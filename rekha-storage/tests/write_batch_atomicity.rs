use rekha_storage::{RocksVectorStore, WriteBatch};
use rekha_core::VectorStoreBackend;

#[test]
fn write_batch_vector_and_payload_atomicity() {
    let dir = tempfile::TempDir::new().unwrap();
    let store = RocksVectorStore::open(dir.path()).unwrap();

    let batch = WriteBatch::new(&store)
        .put_vector(1, 100, &[1.0, 2.0, 3.0])
        .put_payload(1, b"test_payload");
    batch.commit().unwrap();

    let rec = store.get_vector_record(1).unwrap().unwrap();
    assert_eq!(rec.id, 1);
    assert_eq!(rec.timestamp, 100);
    assert!(!rec.is_tombstone);
    assert_eq!(rec.data, Some(vec![1.0, 2.0, 3.0]));

    let payload = store.get_payload(1).unwrap().unwrap();
    assert_eq!(payload, b"test_payload");
}

#[test]
fn write_batch_tombstone_atomicity() {
    let dir = tempfile::TempDir::new().unwrap();
    let store = RocksVectorStore::open(dir.path()).unwrap();

    let batch = WriteBatch::new(&store)
        .put_vector(1, 100, &[10.0])
        .put_tombstone(1, 200);
    batch.commit().unwrap();

    let rec = store.get_vector_record(1).unwrap().unwrap();
    assert!(rec.is_tombstone);
    assert_eq!(rec.timestamp, 200);
    assert!(store.get_vector(1).unwrap().is_none());
}
