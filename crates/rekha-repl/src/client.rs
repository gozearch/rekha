//! HTTP client for RekhaDB API.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct RekhaClient {
    url: String,
    api_key: Option<String>,
    http: reqwest::Client,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Collection {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub dimension: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QueryResult {
    pub ids: Vec<Vec<String>>,
    pub distances: Option<Vec<Vec<f32>>>,
    pub metadatas: Option<Vec<Vec<Option<serde_json::Value>>>>,
    pub documents: Option<Vec<Vec<Option<String>>>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GetResult {
    pub ids: Vec<String>,
    pub embeddings: Option<Vec<Vec<f32>>>,
    pub metadatas: Option<Vec<Option<serde_json::Value>>>,
    pub documents: Option<Vec<Option<String>>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AddRequest {
    pub ids: Vec<String>,
    pub embeddings: Vec<Vec<f32>>,
    pub metadatas: Option<Vec<Option<serde_json::Value>>>,
    pub documents: Option<Vec<Option<String>>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QueryRequest {
    pub query_embeddings: Vec<Vec<f32>>,
    pub n_results: Option<usize>,
    #[serde(rename = "where")]
    pub where_filter: Option<serde_json::Value>,
    pub include: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GetRequest {
    pub ids: Option<Vec<String>>,
    #[serde(rename = "where")]
    pub where_filter: Option<serde_json::Value>,
    pub include: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DeleteRequest {
    pub ids: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateCollectionRequest {
    pub name: String,
    #[serde(default)]
    pub get_or_create: bool,
}

impl RekhaClient {
    pub fn new(url: &str, api_key: Option<&str>) -> Self {
        Self {
            url: url.trim_end_matches('/').to_string(),
            api_key: api_key.map(|s| s.to_string()),
            http: reqwest::Client::new(),
        }
    }

    fn headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        if let Some(ref key) = self.api_key
            && let Ok(val) = key.parse()
        {
            headers.insert("x-chroma-token", val);
        }
        headers
    }

    pub async fn heartbeat(&self) -> Result<u64, String> {
        let resp = self
            .http
            .get(format!("{}/api/v2/heartbeat", self.url))
            .headers(self.headers())
            .send()
            .await
            .map_err(|e| e.to_string())?;
        let body: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
        body["nanosecond heartbeat"]
            .as_u64()
            .ok_or_else(|| "Invalid heartbeat response".into())
    }

    pub async fn list_collections(
        &self,
        tenant: &str,
        database: &str,
    ) -> Result<Vec<Collection>, String> {
        let resp = self
            .http
            .get(format!(
                "{}/api/v2/tenants/{tenant}/databases/{database}/collections",
                self.url
            ))
            .headers(self.headers())
            .send()
            .await
            .map_err(|e| e.to_string())?;
        resp.json().await.map_err(|e| e.to_string())
    }

    pub async fn create_collection(
        &self,
        tenant: &str,
        database: &str,
        name: &str,
    ) -> Result<Collection, String> {
        let resp = self
            .http
            .post(format!(
                "{}/api/v2/tenants/{tenant}/databases/{database}/collections",
                self.url
            ))
            .headers(self.headers())
            .json(&CreateCollectionRequest {
                name: name.to_string(),
                get_or_create: true,
            })
            .send()
            .await
            .map_err(|e| e.to_string())?;
        resp.json().await.map_err(|e| e.to_string())
    }

    pub async fn add(
        &self,
        tenant: &str,
        database: &str,
        collection: &str,
        req: AddRequest,
    ) -> Result<(), String> {
        let resp = self
            .http
            .post(format!(
                "{}/api/v2/tenants/{tenant}/databases/{database}/collections/{collection}/add",
                self.url
            ))
            .headers(self.headers())
            .json(&req)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(format!("Add failed: {}", resp.status()))
        }
    }

    pub async fn query(
        &self,
        tenant: &str,
        database: &str,
        collection: &str,
        req: QueryRequest,
    ) -> Result<QueryResult, String> {
        let resp = self
            .http
            .post(format!(
                "{}/api/v2/tenants/{tenant}/databases/{database}/collections/{collection}/query",
                self.url
            ))
            .headers(self.headers())
            .json(&req)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        resp.json().await.map_err(|e| e.to_string())
    }

    pub async fn get(
        &self,
        tenant: &str,
        database: &str,
        collection: &str,
        req: GetRequest,
    ) -> Result<GetResult, String> {
        let resp = self
            .http
            .post(format!(
                "{}/api/v2/tenants/{tenant}/databases/{database}/collections/{collection}/get",
                self.url
            ))
            .headers(self.headers())
            .json(&req)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        resp.json().await.map_err(|e| e.to_string())
    }

    pub async fn count(
        &self,
        tenant: &str,
        database: &str,
        collection: &str,
    ) -> Result<u64, String> {
        let resp = self
            .http
            .get(format!(
                "{}/api/v2/tenants/{tenant}/databases/{database}/collections/{collection}/count",
                self.url
            ))
            .headers(self.headers())
            .send()
            .await
            .map_err(|e| e.to_string())?;
        let body: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
        body["count"]
            .as_u64()
            .ok_or_else(|| "Invalid count response".into())
    }

    pub async fn delete(
        &self,
        tenant: &str,
        database: &str,
        collection: &str,
        req: DeleteRequest,
    ) -> Result<(), String> {
        let resp = self
            .http
            .post(format!(
                "{}/api/v2/tenants/{tenant}/databases/{database}/collections/{collection}/delete",
                self.url
            ))
            .headers(self.headers())
            .json(&req)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(format!("Delete failed: {}", resp.status()))
        }
    }

    pub async fn delete_collection(
        &self,
        tenant: &str,
        database: &str,
        name: &str,
    ) -> Result<(), String> {
        let resp = self
            .http
            .delete(format!(
                "{}/api/v2/tenants/{tenant}/databases/{database}/collections/{name}",
                self.url
            ))
            .headers(self.headers())
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(format!("Delete collection failed: {}", resp.status()))
        }
    }
}
