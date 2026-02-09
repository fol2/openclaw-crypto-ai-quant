/// Average True Range — Wilder smoothing, matching Python `ta.volatility.average_true_range`.
#[derive(Debug, Clone)]
pub struct AtrIndicator {
    window: usize,
    prev_close: f64,
    atr_value: f64,
    count: usize,
    sum: f64,
    warm: bool,
    has_prev: bool,
}

impl AtrIndicator {
    pub fn new(window: usize) -> Self {
        Self {
            window,
            prev_close: 0.0,
            atr_value: 0.0,
            count: 0,
            sum: 0.0,
            warm: false,
            has_prev: false,
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> f64 {
        let tr = if self.has_prev {
            (high - low)
                .max((high - self.prev_close).abs())
                .max((low - self.prev_close).abs())
        } else {
            high - low
        };

        self.has_prev = true;
        self.prev_close = close;

        if !self.warm {
            self.sum += tr;
            self.count += 1;
            if self.count >= self.window {
                self.atr_value = self.sum / self.window as f64;
                self.warm = true;
            } else {
                self.atr_value = self.sum / self.count as f64;
            }
        } else {
            // Wilder smoothing: ATR = (prev_ATR * (N-1) + TR) / N
            self.atr_value =
                (self.atr_value * (self.window as f64 - 1.0) + tr) / self.window as f64;
        }
        self.atr_value
    }

    pub fn is_warm(&self) -> bool {
        self.warm
    }
}
