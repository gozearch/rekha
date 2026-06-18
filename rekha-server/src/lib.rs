pub mod config;
pub mod coordinator;
pub mod raft_network;
pub mod server;
pub mod service;

pub use config::ServerConfig;
pub use coordinator::Coordinator;
pub use server::ServerInstance;

/// Re-export the generated protobuf types.
pub mod proto {
    tonic::include_proto!("rekha");
}
