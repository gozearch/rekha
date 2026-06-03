pub mod client;

pub use client::RekhaClient;

/// Re-export the generated protobuf types.
pub mod proto {
    tonic::include_proto!("rekha");
}
