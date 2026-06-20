pub mod config;
pub mod coordinator;
pub(crate) mod peer;
pub mod server;
pub mod service;

pub use config::ServerConfig;
pub use coordinator::Coordinator;
pub use server::ServerInstance;

/// Re-export the generated protobuf types.
pub mod proto {
    tonic::include_proto!("rekha");
}
