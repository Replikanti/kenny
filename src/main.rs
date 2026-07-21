use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match kenny::cli::run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(kenny::Error::Usage(msg)) => {
            eprintln!("kenny: {msg}");
            eprintln!("run `kenny --help` for usage");
            ExitCode::from(2)
        }
        Err(e) => {
            eprintln!("kenny: {e}");
            ExitCode::FAILURE
        }
    }
}
