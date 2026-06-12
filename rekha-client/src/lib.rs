pub mod client;

pub use client::{ClientConfig, RekhaClient};

/// Re-export the generated protobuf types.
pub mod proto {
    tonic::include_proto!("rekha");
}
