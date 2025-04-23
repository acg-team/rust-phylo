use rstest::*;

use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use tempfile::tempdir;

use crate::io::{read_sequences, write_newick_to_file, write_sequences_to_file};
use crate::{record_wo_desc as record, tree};

#[test]
fn reading_correct_fasta() {
    let sequences = read_sequences(&PathBuf::from("./data/sequences_DNA1.fasta")).unwrap();
    assert_eq!(sequences.len(), 4);
    for seq in sequences {
        assert_eq!(seq.seq().len(), 5);
    }

    let corr_lengths = [1, 2, 2, 4];
    let sequences =
        read_sequences(&PathBuf::from("./data/sequences_DNA2_unaligned.fasta")).unwrap();
    assert_eq!(sequences.len(), 4);
    for (i, seq) in sequences.into_iter().enumerate() {
        assert_eq!(seq.seq().len(), corr_lengths[i]);
    }
}

#[rstest]
#[case::empty_sequence_name("./data/sequences_garbage_empty_name.fasta", "Expecting id")]
#[case::garbage_sequence(
    "./data/sequences_garbage_non-ascii.fasta",
    "Non-ascii character found"
)]
#[case::weird_chars(
    "./data/sequences_garbage_weird_symbols.fasta",
    "Invalid genetic sequence"
)]
fn reading_incorrect_fasta(#[case] input: &str, #[case] exp_error: &str) {
    let res = read_sequences(&PathBuf::from(input));
    assert!(res.is_err());
    assert!(res.unwrap_err().to_string().contains(exp_error));
}

#[test]
fn reading_nonexistent_fasta() {
    assert!(read_sequences(&PathBuf::from("./data/sequences_nonexistent.fasta")).is_err());
}

#[test]
fn test_write_sequences_to_file() {
    let sequences = vec![record!("seq1", b"ATGC"), record!("seq2", b"CGTA")];
    let temp_dir = tempdir().unwrap();
    let output_path = temp_dir.path().join("output.fasta");
    write_sequences_to_file(&sequences, &output_path).unwrap();
    let mut file_content = String::new();
    std::fs::File::open(output_path)
        .unwrap()
        .read_to_string(&mut file_content)
        .unwrap();
    let expected_output = ">seq1\nATGC\n>seq2\nCGTA\n";
    assert_eq!(file_content, expected_output);
}

#[test]
fn test_write_sequences_to_file_bad_path() {
    let sequences = vec![record!("seq1", b"ATGC"), record!("seq2", b"CGTA")];
    let temp_dir = tempdir().unwrap();
    let output_path = temp_dir
        .path()
        .join("nonexistent_folder")
        .join("output.fasta");
    assert!(write_sequences_to_file(&sequences, &output_path).is_err());
}

#[test]
fn test_write_sequences_to_existing_file() {
    let sequences = vec![record!("seq1", b"ATGC"), record!("seq2", b"CGTA")];
    let temp_dir = tempdir().unwrap();
    let output_path = temp_dir.path().join("output.fasta");
    File::create(&output_path).unwrap();
    assert!(write_sequences_to_file(&sequences, &output_path).is_err());
}

#[test]
fn test_write_newick_to_file() {
    let newick = "(((A:1.4,B:2.45):1,(D:1.2,E:2.1):1):0);";
    let tree = tree!(newick);
    let temp_dir = tempdir().unwrap();
    let output_path = temp_dir.path().join("output.newick");

    write_newick_to_file(&[tree], output_path.clone()).unwrap();

    let mut file_content = String::new();
    std::fs::File::open(output_path)
        .unwrap()
        .read_to_string(&mut file_content)
        .unwrap();
    assert_eq!(file_content.trim(), newick);
}

#[test]
fn test_write_multiple_newick_to_file() {
    let newick0 = "(((((A:1,B:1)F:1,C:2)G:1,D:3)H:1,E:4)I:1);";
    let newick1 = "(((A:1.5,B:2.3)E:5.1,(C:3.9,D:4.8)F:6.2)G:7.3);";
    let newick2 = "((A:1,(B:1,C:1)E:2)F:1);";
    let trees = vec![tree!(newick0), tree!(newick1), tree!(newick2)];

    let temp_dir = tempdir().unwrap();
    let output_path = temp_dir.path().join("output.newick");

    write_newick_to_file(&trees, output_path.clone()).unwrap();

    let mut file_content = String::new();
    std::fs::File::open(output_path)
        .unwrap()
        .read_to_string(&mut file_content)
        .unwrap();
    assert_eq!(
        file_content.trim(),
        format!("{}\n{}\n{}", newick0, newick1, newick2)
    );
}

#[test]
fn test_write_newickto_file_bad_path() {
    let tree = tree!("(((A:1.4,B:2.45):1,(D:1.2,E:2.1):1):0);");

    let temp_dir = tempdir().unwrap();
    let output_path = temp_dir
        .path()
        .join("nonexistent_folder")
        .join("output.newick");
    assert!(write_newick_to_file(&[tree], output_path).is_err());
}

#[test]
fn test_write_newick_to_existing_file() {
    let tree = tree!("(((A:1.4,B:2.45):1,(D:1.2,E:2.1):1):0);");
    let temp_dir = tempdir().unwrap();
    let output_path = temp_dir.path().join("output.newick");
    File::create(&output_path).unwrap();
    assert!(write_newick_to_file(&[tree], output_path).is_err());
}

#[test]
fn read_sequences_weird_gap_chars() {
    let sequences_underscore =
        read_sequences(&PathBuf::from("./data/sequences_gap_underscore.fasta")).unwrap();
    let sequences_asterisk =
        read_sequences(&PathBuf::from("./data/sequences_gap_asterisk.fasta")).unwrap();
    let sequences = read_sequences(&PathBuf::from("./data/sequences_gap_normal.fasta")).unwrap();

    assert_eq!(sequences.len(), 4);
    assert_eq!(sequences_underscore.len(), sequences.len());
    assert_eq!(sequences_asterisk.len(), sequences.len());

    for (i, seq) in sequences.into_iter().enumerate() {
        assert_eq!(seq.seq().len(), 8);
        assert_eq!(seq.seq(), sequences_underscore[i].seq());
        assert_eq!(seq.seq(), sequences_asterisk[i].seq());
    }
}
