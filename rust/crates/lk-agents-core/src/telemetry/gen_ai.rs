//! OpenTelemetry GenAI semantic conventions the framework's spans follow.

/// `gen_ai_provider_name` (`trace_types.py`): a plugin's `provider` in the GenAI registry's
/// spelling; one outside the registry keeps its own.
pub fn provider_name(provider: &str) -> Option<String> {
    let value = provider.trim();
    if value.is_empty() {
        return None;
    }
    let host = value.to_lowercase();
    let by_host = match host.as_str() {
        "api.anthropic.com" => Some("anthropic"),
        "api.cohere.ai" | "api.cohere.com" => Some("cohere"),
        "api.deepseek.com" => Some("deepseek"),
        "api.groq.com" => Some("groq"),
        "api.mistral.ai" => Some("mistral_ai"),
        "api.moonshot.ai" | "api.moonshot.cn" => Some("moonshot_ai"),
        "api.openai.com" => Some("openai"),
        "api.perplexity.ai" => Some("perplexity"),
        "api.x.ai" => Some("x_ai"),
        "generativelanguage.googleapis.com" => Some("gcp.gemini"),
        _ => None,
    };
    let by_suffix = || {
        [
            (".openai.azure.com", "azure.ai.openai"),
            (".services.ai.azure.com", "azure.ai.inference"),
            (".aiplatform.googleapis.com", "gcp.vertex_ai"),
        ]
        .into_iter()
        .find(|(suffix, _)| host.ends_with(suffix))
        .map(|(_, mapped)| mapped)
    };
    if let Some(mapped) = by_host.or_else(by_suffix) {
        return Some(mapped.to_owned());
    }
    // Only the Bedrock endpoints, not every AWS service that shares the domain.
    if host.starts_with("bedrock") && host.ends_with(".amazonaws.com") {
        return Some("aws.bedrock".to_owned());
    }
    let canonical: String = host.chars().filter(|c| c.is_alphanumeric()).collect();
    let by_name = match canonical.as_str() {
        "amazon" | "amazonbedrock" | "awsbedrock" | "bedrock" => "aws.bedrock",
        "anthropic" => "anthropic",
        "azureaiinference" => "azure.ai.inference",
        "azureopenai" => "azure.ai.openai",
        "cohere" => "cohere",
        "deepseek" => "deepseek",
        "gemini" => "gcp.gemini",
        "google" | "googlecloudplatform" | "googlegenai" => "gcp.gen_ai",
        "groq" => "groq",
        "ibmwatsonxai" | "watsonx" => "ibm.watsonx.ai",
        "mistral" | "mistralai" => "mistral_ai",
        "moonshot" | "moonshotai" => "moonshot_ai",
        "openai" => "openai",
        "perplexity" => "perplexity",
        "vertexai" | "vertexaimodelgarden" => "gcp.vertex_ai",
        "xai" => "x_ai",
        _ => return Some(value.to_owned()),
    };
    Some(by_name.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: a registry spelling lost, or an unknown provider dropped instead of kept.
    #[test]
    fn provider_names_follow_the_genai_registry() {
        assert_eq!(provider_name("api.openai.com").as_deref(), Some("openai"));
        assert_eq!(provider_name("my.openai.azure.com").as_deref(), Some("azure.ai.openai"));
        let bedrock = provider_name("bedrock-runtime.us-east-1.amazonaws.com");
        assert_eq!(bedrock.as_deref(), Some("aws.bedrock"));
        assert_eq!(provider_name("Google GenAI").as_deref(), Some("gcp.gen_ai"));
        assert_eq!(provider_name("livekit").as_deref(), Some("livekit"));
        assert_eq!(provider_name("  "), None);
    }
}
