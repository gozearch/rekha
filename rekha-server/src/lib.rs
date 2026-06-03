pub mod config;
pub mod coordinator;
pub mod service;
pub mod server;

pub use config::ServerConfig;
pub use coordinator::Coordinator;
pub use server::ServerInstance;

/// Re-export the generated protobuf types.
pub mod proto {
    tonic::include_proto!("rekha");
}
