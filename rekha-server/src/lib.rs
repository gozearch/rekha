pub mod config;
pub mod coordinator;
pub mod server;
pub mod service;

pub use config::ServerConfig;
pub use coordinator::Coordinator;
pub use server::ServerInstance;

pub mod proto {
    tonic::include_proto!("rekha");
}
