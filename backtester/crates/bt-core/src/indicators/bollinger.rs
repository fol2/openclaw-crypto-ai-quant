use super::RingBuf;

/// Bollinger Bands — rolling SMA + population std dev (ddof=0).
/// Matches Python `ta.volatility.BollingerBands`.
#[derive(Debug, Clone)]
pub struct BollingerBands {
    ring: RingBuf,
    num_std: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct BbOutput {
    pub upper: f64,
    pub middle: f64,
    pub lower: f64,
}

impl BollingerBands {
    pub fn new(window: usize) -> Self {
        Self {
            ring: RingBuf::new(window),
            num_std: 2.0,
        }
    }

    pub fn update(&mut self, close: f64) -> BbOutput {
        self.ring.push(close);
        if !self.ring.full() {
            let mean = self.ring.mean();
            return BbOutput {
                upper: mean,
                middle: mean,
                lower: mean,
            };
        }
        let middle = self.ring.mean();
        let std = self.ring.std_pop();
        BbOutput {
            upper: middle + self.num_std * std,
            middle,
            lower: middle - self.num_std * std,
        }
    }
}
