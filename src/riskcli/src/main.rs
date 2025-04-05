// CLI Risk calculation
// Loads a TOML file and prints the risk output
use std::io::Read;
use std::fs;
use std::env;

use sunlib::risk::RiskModel;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() == 2 && args[1] == "--help" {
        println!("Suncal Decision Risk Calculator");
        std::process::exit(0);
    }

    let tstr: String = if args.len() < 2 {
        let stdin = std::io::stdin();
        let mut input = String::new();
        let mut handle = stdin.lock();
        match handle.read_to_string(&mut input) {
            Ok(_) => input,
            Err(e) => { eprintln!("stdin {}", e.to_string()); std::process::exit(0); },
        }
    } else {
        match fs::read_to_string(&args[1]) {
            Ok(v) => v,
            Err(e) => { eprintln!("file {}, {}", args[1], e.to_string()); std::process::exit(0); },
        }
    };

    let risk = match RiskModel::load_toml(&tstr) {
        Ok(v) => v,
        Err(e) => { eprintln!("{}", e.to_string()); std::process::exit(0); },
    };

    let result = risk.calculate();
    println!("{}", result.to_string());
}
