// Suncal CLI - Loads TOML file and prints output
use std::{error::Error, fs, env};
use toml;

use units;
use sunlib::cfg::MeasureSystem;


fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: [configfile] ");
        std::process::exit(0);
    }

    let config = match fs::read_to_string(&args[1]) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("filename {}, {}", args[1], e.to_string());
            std::process::exit(0);
        }
    };

    units::init();

    let model: MeasureSystem = match toml::from_str(&config) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{}", e.to_string());
            std::process::exit(0);
        }
    };

    const STACKSIZE: usize = 8;  // How big should the stack be??
    std::thread::Builder::new().stack_size(1024*1024*STACKSIZE).spawn(move || {
        let result = model.calculate();
        match result {
            Ok(r) => r.printit(),
            Err(r) => eprintln!("{}", r),
        }
    })?.join().unwrap();

    Ok(())
}
