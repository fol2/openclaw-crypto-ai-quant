#[cfg(test)]
use std::env;
#[cfg(test)]
use std::sync::{Mutex, OnceLock};

#[cfg(test)]
static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

#[cfg(test)]
pub(crate) struct EnvGuard {
    saved: Vec<(&'static str, Option<std::ffi::OsString>)>,
}

#[cfg(test)]
impl EnvGuard {
    pub(crate) fn set(vars: &[(&'static str, Option<&str>)]) -> Self {
        let mut saved = Vec::new();
        for (name, value) in vars {
            saved.push((*name, env::var_os(name)));
            match value {
                Some(value) => env::set_var(name, value),
                None => env::remove_var(name),
            }
        }
        Self { saved }
    }
}

#[cfg(test)]
impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (name, value) in self.saved.drain(..) {
            match value {
                Some(value) => env::set_var(name, value),
                None => env::remove_var(name),
            }
        }
    }
}

#[cfg(test)]
pub(crate) fn env_lock() -> &'static Mutex<()> {
    ENV_LOCK.get_or_init(|| Mutex::new(()))
}
