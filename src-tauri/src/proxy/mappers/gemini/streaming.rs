// Gemini 原生协议的发散适配器
//
// 只负责“v1internal SSE 行 → 客户端 Gemini SSE 行”的协议差异：
// 签名缓存、工具 ID 注入、图片统计、v1internal 解包、`__cloudCodeMeta` 首帧。
// 心跳、首字/状态切换/滑动超时、优雅收尾与首包错误契约全部由
// `pipeline::stream_lifecycle::run_stream_lifecycle` 统一驱动。

use bytes::Bytes;
use serde_json::Value;
use std::sync::Arc;

use crate::proxy::mappers::error_classifier::StreamErrorReport;
use crate::proxy::pipeline::stream_lifecycle::ProtocolStreamHandler;
use crate::proxy::thinking_store::TurnAccumulator;

pub struct GeminiStreamHandler {
    cloud_code_trace_id: Option<String>,
    session_id: String,
    model_name: String,
    message_count: usize,
    track_image_success: bool,
    image_success_manager: Option<Arc<crate::proxy::TokenManager>>,
    image_success_account: String,
    image_success_model: String,
    thinking_acc: Option<TurnAccumulator>,
    saw_image_data: bool,
    has_content: bool,
    has_thinking: bool,
}

impl GeminiStreamHandler {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cloud_code_trace_id: Option<String>,
        session_id: String,
        model_name: String,
        message_count: usize,
        causal_anchor: String,
        track_image_success: bool,
        image_success_manager: Option<Arc<crate::proxy::TokenManager>>,
        image_success_account: String,
        image_success_model: String,
    ) -> Self {
        Self {
            cloud_code_trace_id,
            session_id,
            model_name,
            message_count,
            track_image_success,
            image_success_manager,
            image_success_account,
            image_success_model,
            thinking_acc: Some(TurnAccumulator::with_anchor(causal_anchor)),
            saw_image_data: false,
            has_content: false,
            has_thinking: false,
        }
    }
}

impl ProtocolStreamHandler for GeminiStreamHandler {
    fn initial_frames(&mut self) -> Vec<Bytes> {
        // 官方 Worker 会将 TraceID 作为 SSE 流的第 0 个数据包下发
        match &self.cloud_code_trace_id {
            Some(tid) => {
                let meta_pkg = serde_json::json!({ "__cloudCodeMeta": { "traceId": tid } });
                vec![Bytes::from(format!(
                    "data: {}\n\n",
                    serde_json::to_string(&meta_pkg).unwrap_or_default()
                ))]
            }
            None => Vec::new(),
        }
    }

    fn heartbeat_frame(&mut self) -> Bytes {
        // [对齐 Gemini 官方端点规范] 官方流从不携带 SSE 注释帧（`: ping`），
        // 而 google-genai-sdk（Antigravity CLI / agy）将任何非 `data:` 行判为
        // `invalid stream chunk` 并整轮判失败。此处改用与官方同形的空 candidates 续帧保活。
        Bytes::from(
            "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[]},\"index\":0}]}\n\n",
        )
    }

    fn process_line(&mut self, line: &str) -> Vec<Bytes> {
        // 注释行 / 非数据行一律丢弃：官方端点不下发此类帧，
        // 透传会让严格解析器（google-genai-sdk）判定流非法。
        // 用 `data:` 前缀而非 `data: `，兼容上游省略空格的形式。
        let Some(rest) = line.strip_prefix("data:") else {
            return Vec::new();
        };

        let json_part = rest.trim();
        if json_part == "[DONE]" {
            return vec![Bytes::from("data: [DONE]\n\n")];
        }

        let mut json: Value = match serde_json::from_str(json_part) {
            Ok(v) => v,
            Err(e) => {
                let error_json = serde_json::json!({
                    "error": {
                        "code": 502,
                        "message": format!("Gemini stream json parse error: {e}"),
                        "status": "INTERNAL"
                    }
                });
                return vec![Bytes::from(format!(
                    "data: {}\n\n",
                    serde_json::to_string(&error_json).unwrap_or_default()
                ))];
            }
        };

        if self.track_image_success && crate::proxy::pipeline::response_has_inline_image_data(&json)
        {
            self.saw_image_data = true;
        }

        // v1internal 响应可能包一层 "response"
        let parts_snapshot = json
            .get("response")
            .unwrap_or(&json)
            .get("candidates")
            .and_then(|c| c.as_array())
            .cloned()
            .unwrap_or_default();

        for cand in &parts_snapshot {
            if let Some(parts) = cand
                .get("content")
                .and_then(|c| c.get("parts"))
                .and_then(|p| p.as_array())
            {
                for part in parts {
                    if part.get("text").is_some()
                        || part.get("functionCall").is_some()
                        || part.get("inlineData").is_some()
                        || part.get("inline_data").is_some()
                    {
                        self.has_content = true;
                    }
                    if part.get("thought").and_then(|v| v.as_bool()) == Some(true) {
                        self.has_thinking = true;
                    }
                    if let Some(acc) = self.thinking_acc.as_mut() {
                        acc.ingest_part(part);
                    }
                    if let Some(sig) = part.get("thoughtSignature").and_then(|s| s.as_str()) {
                        crate::proxy::SignatureCache::global().cache_session_signature(
                            &self.session_id,
                            sig.to_string(),
                            self.message_count,
                        );
                        tracing::debug!(
                            "[Gemini-SSE] Cached session signature (len: {}) for session: {} at count {}",
                            sig.len(),
                            self.session_id,
                            self.message_count
                        );
                    }
                }
            }
        }

        // 注入工具调用 ID
        crate::proxy::mappers::gemini::wrapper::inject_ids_to_response(&mut json, &self.model_name);

        // 解包 v1internal 信封
        if let Some(inner) = json.get_mut("response").map(|v| v.take()) {
            vec![Bytes::from(format!(
                "data: {}\n\n",
                serde_json::to_string(&inner).unwrap_or_default()
            ))]
        } else {
            vec![Bytes::from(format!(
                "data: {}\n\n",
                serde_json::to_string(&json).unwrap_or_default()
            ))]
        }
    }

    fn has_content(&self) -> bool {
        self.has_content
    }

    fn has_thinking(&self) -> bool {
        self.has_thinking
    }

    fn emit_finalize(&mut self) -> Vec<Bytes> {
        // 图片生成成功统计仅在优雅结束时提交（错误路径不会走到这里）
        if self.track_image_success && self.saw_image_data {
            if let Some(mgr) = &self.image_success_manager {
                mgr.mark_account_success(&self.image_success_account);
                mgr.clear_persisted_live_limit(
                    &self.image_success_account,
                    Some(&self.image_success_model),
                );
            }
        }
        // Gemini 正常 EOF 不额外下发 [DONE]，保持既有行为
        Vec::new()
    }

    fn emit_initial_error(&mut self, error_report: &StreamErrorReport) -> Vec<Bytes> {
        let error_json = serde_json::json!({
            "error": {
                "code": 503,
                "message": error_report.client_message(),
                "status": "UNAVAILABLE",
                "function": error_report.function,
                "call_site": error_report.call_site(),
                "params": error_report.params
            }
        });
        vec![
            Bytes::from(format!(
                "data: {}\n\n",
                serde_json::to_string(&error_json).unwrap_or_default()
            )),
            Bytes::from("data: [DONE]\n\n"),
        ]
    }

    fn on_finish(&mut self) {
        // 与 Claude/OpenAI 一致：流结束（含优雅收尾）时提交思维链
        if let Some(acc) = self.thinking_acc.take() {
            acc.commit(&self.session_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proxy::pipeline::stream_lifecycle::ProtocolStreamHandler;

    fn handler() -> GeminiStreamHandler {
        GeminiStreamHandler::new(
            None,
            "sess".to_string(),
            "gemini-3.8-flash".to_string(),
            1,
            "anchor".to_string(),
            false,
            None,
            String::new(),
            String::new(),
        )
    }

    #[test]
    fn heartbeat_frame_is_a_data_frame_not_an_sse_comment() {
        let frame = handler().heartbeat_frame();
        let text = String::from_utf8(frame.to_vec()).unwrap();
        assert!(text.starts_with("data: "));
        assert!(!text.contains(": ping"));
        let json: Value =
            serde_json::from_str(text.trim_start_matches("data: ").trim()).expect("valid json");
        assert_eq!(
            json["candidates"][0]["content"]["parts"],
            serde_json::json!([])
        );
    }

    #[test]
    fn non_data_lines_are_dropped() {
        let mut h = handler();
        assert!(h.process_line(": ping").is_empty());
        assert!(h.process_line("event: message").is_empty());
        assert!(h.process_line("id: 1").is_empty());
    }

    #[test]
    fn data_prefix_without_space_is_accepted() {
        let mut h = handler();
        let out = h.process_line(r#"data:{"candidates":[{"content":{"parts":[{"text":"hi"}]}}]}"#);
        assert_eq!(out.len(), 1);
        let text = String::from_utf8(out[0].to_vec()).unwrap();
        assert!(text.contains("\"hi\""));
    }

    #[test]
    fn done_sentinel_still_forwarded() {
        let mut h = handler();
        let out = h.process_line("data: [DONE]");
        assert_eq!(out.len(), 1);
        assert_eq!(&out[0][..], b"data: [DONE]\n\n");
    }
}
