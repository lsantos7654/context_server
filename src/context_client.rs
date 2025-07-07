use anyhow::{anyhow, Result};
use reqwest::Client;
use serde::Serialize;

use crate::{CodeSnippet, SearchResponse};

#[derive(Clone)]
pub struct ContextClient {
    client: Client,
    base_url: String,
}

impl Default for ContextClient {
    fn default() -> Self {
        Self::new("http://localhost:8000")
    }
}

impl ContextClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.to_string(),
        }
    }

    pub async fn search(
        &self,
        context_name: &str,
        query: &str,
        limit: usize,
    ) -> Result<SearchResponse> {
        let url = format!("{}/api/contexts/{}/search", self.base_url, context_name);
        
        let request_body = SearchRequest {
            query: query.to_string(),
            mode: "hybrid".to_string(),
            limit,
        };

        let response = self
            .client
            .post(&url)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| anyhow!("Failed to send search request: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(anyhow!(
                "Search request failed with status {}: {}",
                status,
                text
            ));
        }

        let search_response: SearchResponse = response
            .json()
            .await
            .map_err(|e| anyhow!("Failed to parse search response: {}", e))?;

        Ok(search_response)
    }

    pub async fn get_code_snippet(
        &self,
        context_name: &str,
        snippet_id: &str,
    ) -> Result<CodeSnippet> {
        let url = format!(
            "{}/api/contexts/{}/code-snippets/{}",
            self.base_url, context_name, snippet_id
        );

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| anyhow!("Failed to send code snippet request: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(anyhow!(
                "Code snippet request failed with status {}: {}",
                status,
                text
            ));
        }

        let snippet: CodeSnippet = response
            .json()
            .await
            .map_err(|e| anyhow!("Failed to parse code snippet response: {}", e))?;

        Ok(snippet)
    }
}

#[derive(Debug, Serialize)]
struct SearchRequest {
    query: String,
    mode: String,
    limit: usize,
}