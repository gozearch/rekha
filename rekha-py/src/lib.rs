use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rekha_client::RekhaClient;

/// Python SDK for the Rekha distributed vector database.
///
/// Usage:
/// ```python
/// import rekha
///
/// # Connect to cluster
/// client = rekha.connect("localhost:50051")
///
/// # Insert a vector
/// client.insert(42, [0.1, 0.2, ..., 0.768])
///
/// # Search
/// results = client.search([0.1, 0.2, ..., 0.768], top_k=10)
/// for r in results:
///     print(r.id, r.score)
///
/// # Bulk insert
/// client.insert_batch(vectors_iter)
/// ```

/// A Python-accessible ScoredPoint.
#[pyclass(name = "ScoredPoint")]
#[derive(Debug, Clone)]
struct PyScoredPoint {
    #[pyo3(get)]
    id: u64,
    #[pyo3(get)]
    score: f32,
    #[pyo3(get)]
    payload: Option<Vec<u8>>,
}

/// A Python-accessible client handle.
#[pyclass(name = "Client")]
struct PyClient {
    inner: RekhaClient,
}

#[pymethods]
impl PyClient {
    /// Insert a vector with optional payload.
    fn insert(
        &self,
        id: u64,
        vector: Vec<f32>,
        payload: Option<Vec<u8>>,
    ) -> PyResult<()> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        rt.block_on(self.inner.insert(id, vector, payload))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Search for nearest neighbors.
    fn search(&self, query: Vec<f32>, top_k: usize) -> PyResult<Vec<PyScoredPoint>> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let results = rt
            .block_on(self.inner.search(query, top_k))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(results
            .into_iter()
            .map(|r| PyScoredPoint {
                id: r.id,
                score: r.score,
                payload: r.payload.map(|p| p.data),
            })
            .collect())
    }

    /// Delete vectors by ID.
    fn delete(&self, ids: Vec<u64>) -> PyResult<u64> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        rt.block_on(self.inner.delete(&ids))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

/// Connect to a Rekha cluster.
///
/// Args:
///     address: Address of any seed node (e.g., "localhost:50051")
///
/// Returns:
///     A Client handle.
#[pyfunction]
fn connect(address: &str) -> PyResult<PyClient> {
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let client = rt
        .block_on(RekhaClient::connect(&[address.to_string()]))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok(PyClient { inner: client })
}

/// The rekha Python module.
#[pymodule]
fn rekha(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(connect, m)?)?;
    m.add_class::<PyClient>()?;
    m.add_class::<PyScoredPoint>()?;
    Ok(())
}
