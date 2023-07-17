use std::{env, fs::{self, File}, io::{BufWriter, Write}, time::Instant};

use rmq::*;

mod rmq;
mod y_fast_trie;

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 3 {
        println!("Incorrect arguments");
        return;
    }

    let mode = &args[0];
    let input = &args[1];
    let output = &args[2];

    // read input
    let mut input_data: Vec<u64> = vec![];
    let mut rmq_queries: Vec<(usize, usize)> = vec![];
    for line in fs::read_to_string(input).unwrap().split('\n') {
        if line.is_empty() {
            continue;
        }

        if line.contains(',') {
            let x = line.split_once(',').unwrap();
            rmq_queries.push((x.0.parse().unwrap(), x.1.parse().unwrap()));
        } else {
            input_data.push(line.parse().unwrap());
        }
    }
    let mut output_file = BufWriter::new(File::create(output).unwrap());

    let start_time = Instant::now();
    let space;
    if mode == "rmq" {
        let rmq = LinearRMQ::build(input_data[1..].to_vec());
        space = rmq.bits();
        for (a, b) in rmq_queries {
            let result = rmq.query(a, b);
            write!(output_file, "{}\n", result).unwrap();
        }
    } else if mode == "pd" {
        space = 5;
    } else {
        panic!("invalid mode");
    }
    let end_time = Instant::now();
    println!("RESULT algo={} name=arne_keller time={} space={space}", mode, end_time.duration_since(start_time).as_millis());
}
