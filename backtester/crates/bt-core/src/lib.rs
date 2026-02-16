#![recursion_limit = "256"]

pub mod candle;
pub mod config;
pub mod indicators;
pub mod accounting;
pub mod signals;
pub mod exits;
pub mod position;
pub mod engine;
pub mod decision_kernel;
pub mod kernel_entries;
pub mod kernel_exits;
pub mod state_store;
pub mod init_state;
pub mod sweep;
pub mod reason_codes;
pub mod report;
