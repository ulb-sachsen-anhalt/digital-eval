use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod frame;

use frame::filter_frame;

/// OCR utilities
#[derive(Parser, Debug)]
#[command(name = "ocr-util")]
#[command(version = "1.9.1")]
#[command(about = "OCR Utilities", long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Filter a custom area from an ALTO file
    Frame {
        /// Input ALTO file path
        #[arg(short, long)]
        input: PathBuf,

        /// Output ALTO file path
        #[arg(short, long)]
        output: PathBuf,

        /// Points defining the area to filter (format: "x1,y1 x2,y2 x3,y3 ...")
        #[arg(short, long)]
        points: String,
    },
}

fn main() -> Result<()> {
    let args = Args::parse();

    match args.command {
        Commands::Frame { input, output, points } => {
            filter_frame(&input, &output, &points)?;
            println!("Filtered area saved to: {}", output.display());
        }
    }

    Ok(())
}
