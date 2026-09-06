use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

/// Parses JSON floats from an array key (e.g., "epoch_losses": [2.38, 2.24, ...])
fn extract_json_float_array(json_str: &str, array_key: &str) -> Vec<f64> {
    let mut values = Vec::new();
    if let Some(pos) = json_str.find(array_key) {
        let after_key = &json_str[pos + array_key.len()..];
        if let Some(start_bracket) = after_key.find('[') {
            if let Some(end_bracket) = after_key.find(']') {
                let inner = &after_key[start_bracket + 1..end_bracket];
                for token in inner.split(',') {
                    let cleaned = token.trim().trim_matches('"');
                    if let Ok(val) = cleaned.parse::<f64>() {
                        values.push(val);
                    }
                }
            }
        }
    }
    values
}

/// Parses simple string or numeric key from JSON
fn extract_json_val(json_str: &str, key: &str) -> String {
    if let Some(pos) = json_str.find(key) {
        let after = &json_str[pos + key.len()..];
        let trimmed = after.trim_start_matches(|c: char| c == ':' || c.is_whitespace() || c == '"');
        let val_end = trimmed.find(|c: char| c == ',' || c == '\n' || c == '}' || c == '"').unwrap_or(trimmed.len());
        return trimmed[..val_end].trim().to_string();
    }
    "N/A".to_string()
}

/// Renders a terminal ASCII line graph
fn render_ascii_sparkline(values: &[f64], height: usize, width: usize) -> String {
    if values.is_empty() {
        return "No data points available".to_string();
    }

    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = if (max - min).abs() < 1e-6 { 1.0 } else { max - min };

    let mut output = String::new();
    let sample_step = (values.len() as f64) / (width as f64);

    let sampled: Vec<f64> = (0..width)
        .map(|i| {
            let idx = ((i as f64) * sample_step) as usize;
            values[idx.min(values.len() - 1)]
        })
        .collect();

    for row in (0..=height).rev() {
        let threshold = min + (row as f64 / height as f64) * range;
        output.push_str(&format!("{:6.2} | ", threshold));

        for &val in &sampled {
            let normalized = (val - min) / range * (height as f64);
            if (normalized - row as f64).abs() < 0.6 {
                output.push('●');
            } else if normalized > row as f64 {
                output.push('│');
            } else {
                output.push(' ');
            }
        }
        output.push('\n');
    }

    output.push_str("       +");
    for _ in 0..width {
        output.push('-');
    }
    output.push_str("\n        Epoch 1");
    for _ in 0..(width.saturating_sub(18)) {
        output.push(' ');
    }
    output.push_str(&format!("Epoch {}\n", values.len()));

    output
}

/// Generates a standalone self-contained HTML/SVG visual dashboard
fn generate_html_report(
    losses: &[f64],
    epochs: &str,
    batch_size: &str,
    transitions: &str,
    output_path: &Path,
) -> std::io::Result<()> {
    let mut file = File::create(output_path)?;

    let min_loss = losses.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_loss = losses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let delta = max_loss - min_loss;

    // Build SVG polyline points
    let svg_w = 700.0;
    let svg_h = 280.0;
    let pad_x = 60.0;
    let pad_y = 30.0;
    let plot_w = svg_w - 2.0 * pad_x;
    let plot_h = svg_h - 2.0 * pad_y;

    let mut points = String::new();
    let mut dots = String::new();

    if !losses.is_empty() {
        let n = losses.len();
        for (i, &loss) in losses.iter().enumerate() {
            let x = pad_x + (i as f64 / (n - 1).max(1) as f64) * plot_w;
            let norm_y = if delta.abs() < 1e-6 { 0.5 } else { (loss - min_loss) / delta };
            let y = pad_y + (1.0 - norm_y) * plot_h;

            if i == 0 {
                points.push_str(&format!("{:.1},{:.1}", x, y));
            } else {
                points.push_str(&format!(" {:.1},{:.1}", x, y));
            }

            dots.push_str(&format!(
                "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"4\" fill=\"#38bdf8\"><title>Epoch {}: {:.4}</title></circle>",
                x, y, i + 1, loss
            ));
        }
    }

    let color_axis = "#64748b";
    let color_stroke = "#38bdf8";
    let html_content = format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CNN Loss & Training Progression Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 32px; }}
  .container {{ max-width: 900px; margin: 0 auto; background: #1e293b; border-radius: 12px; padding: 24px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }}
  h1 {{ font-size: 22px; color: #38bdf8; margin-top: 0; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }}
  .stat-card {{ background: #0f172a; padding: 14px; border-radius: 8px; border-left: 4px solid #38bdf8; }}
  .stat-label {{ font-size: 12px; color: #94a3b8; text-transform: uppercase; }}
  .stat-value {{ font-size: 18px; font-weight: bold; color: #f1f5f9; margin-top: 4px; }}
  .chart-box {{ background: #0f172a; border-radius: 8px; padding: 16px; margin-top: 16px; }}
  svg {{ width: 100%; height: auto; display: block; }}
  .grid-line {{ stroke: #334155; stroke-dasharray: 4; stroke-width: 1; }}
  .axis-label {{ fill: #94a3b8; font-size: 11px; font-family: monospace; }}
</style>
</head>
<body>
<div class="container">
  <h1>Deterministic Report: CNN Behavior Cloning Loss</h1>
  <div class="stats-grid">
    <div class="stat-card"><div class="stat-label">Total Epochs</div><div class="stat-value">{epochs}</div></div>
    <div class="stat-card"><div class="stat-label">Batch Size</div><div class="stat-value">{batch_size}</div></div>
    <div class="stat-card"><div class="stat-label">Transitions</div><div class="stat-value">{transitions}</div></div>
    <div class="stat-card"><div class="stat-label">Loss Convergence</div><div class="stat-value">{max_loss:.3} &rarr; {min_loss:.3}</div></div>
  </div>

  <div class="chart-box">
    <svg viewBox="0 0 700 280">
      <!-- Grid lines -->
      <line x1="60" y1="30" x2="640" y2="30" class="grid-line" />
      <line x1="60" y1="100" x2="640" y2="100" class="grid-line" />
      <line x1="60" y1="170" x2="640" y2="170" class="grid-line" />
      <line x1="60" y1="250" x2="640" y2="250" stroke="{color_axis}" stroke-width="1.5" />
      <line x1="60" y1="30" x2="60" y2="250" stroke="{color_axis}" stroke-width="1.5" />

      <!-- Y Axis labels -->
      <text x="50" y="35" text-anchor="end" class="axis-label">{max_loss:.2}</text>
      <text x="50" y="145" text-anchor="end" class="axis-label">{mid_loss:.2}</text>
      <text x="50" y="255" text-anchor="end" class="axis-label">{min_loss:.2}</text>

      <!-- Loss Curve -->
      <polyline fill="none" stroke="{color_stroke}" stroke-width="3" points="{points}" />
      {dots}
    </svg>
  </div>
</div>
</body>
</html>
"#,
        epochs = epochs,
        batch_size = batch_size,
        transitions = transitions,
        max_loss = max_loss,
        min_loss = min_loss,
        mid_loss = (max_loss + min_loss) / 2.0,
        color_axis = color_axis,
        color_stroke = color_stroke,
        points = points,
        dots = dots
    );

    file.write_all(html_content.as_bytes())
}

fn main() {
    let metrics_path = Path::new("/Users/sweeden/kagg/datasets/scottweeden/self-training-code/training_artifacts/metrics/bc_pretrain.json");
    let html_out_path = Path::new("/Users/sweeden/kagg/.antigravity/cnn_training_report.html");

    println!("============================================================");
    println!("    CNN BEHAVIOR CLONING & TRAINING REPORT (RUST ENGINE)   ");
    println!("============================================================");

    if !metrics_path.exists() {
        eprintln!("Error: Metrics file not found at {:?}", metrics_path);
        std::process::exit(1);
    }

    let content = fs::read_to_string(metrics_path).expect("Failed to read bc_pretrain.json");
    let losses = extract_json_float_array(&content, "\"epoch_losses\"");
    let epochs = extract_json_val(&content, "\"epochs\"");
    let batch_size = extract_json_val(&content, "\"batch_size\"");
    let transitions = extract_json_val(&content, "\"bootstrap_transitions\"");

    println!("• Epochs Configured   : {}", epochs);
    println!("• Batch Size          : {}", batch_size);
    println!("• Bootstrap Samples   : {}", transitions);
    println!("• Loss Points Loaded  : {}", losses.len());

    if !losses.is_empty() {
        let initial_loss = losses.first().unwrap();
        let final_loss = losses.last().unwrap();
        let reduction = (initial_loss - final_loss) / initial_loss * 100.0;

        println!("• Initial BC Loss     : {:.4}", initial_loss);
        println!("• Final BC Loss       : {:.4}", final_loss);
        println!("• Total Loss Drop     : {:.2}%", reduction);

        println!("\n--- Terminal Visual: CNN Loss Convergence Sparkline ---");
        let chart = render_ascii_sparkline(&losses, 10, 50);
        println!("{}", chart);
    }

    match generate_html_report(&losses, &epochs, &batch_size, &transitions, html_out_path) {
        Ok(_) => println!("\n[OK] HTML visual report generated: {:?}", html_out_path),
        Err(e) => eprintln!("Error generating visual report: {}", e),
    }
}
