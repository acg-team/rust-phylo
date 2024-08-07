use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use bio::io::fasta;
use rstest::*;
use tempfile::tempdir;

use crate::{
    io::{read_sequences_from_file, write_newick_to_file, write_sequences_to_file},
    tree::tree_parser::from_newick_string,
};

#[test]
fn reading_correct_fasta() {
    let sequences =
        read_sequences_from_file(&PathBuf::from("./data/sequences_DNA1.fasta")).unwrap();
    assert_eq!(sequences.len(), 4);
    for seq in sequences {
        assert_eq!(seq.seq().len(), 5);
    }

    let corr_lengths = [1, 2, 2, 4];
    let sequences =
        read_sequences_from_file(&PathBuf::from("./data/sequences_DNA2_unaligned.fasta")).unwrap();
    assert_eq!(sequences.len(), 4);
    for (i, seq) in sequences.into_iter().enumerate() {
        assert_eq!(seq.seq().len(), corr_lengths[i]);
    }
}

#[rstest]
#[case::empty_sequence_name("./data/sequences_garbage_empty_name.fasta")]
#[case::garbage_sequence("./data/sequences_garbage_non-ascii.fasta")]
#[case::weird_chars("./data/sequences_garbage_weird_symbols.fasta")]
fn reading_incorrect_fasta(#[case] input: &str) {
    assert!(read_sequences_from_file(&PathBuf::from(input)).is_err());
}

#[test]
fn reading_nonexistent_fasta() {
    assert!(
        read_sequences_from_file(&PathBuf::from("./data/sequences_nonexistent.fasta")).is_err()
    );
}

#[test]
fn test_write_sequences_to_file() {
    let sequences = vec![
        fasta::Record::with_attrs("seq1", None, b"ATGC"),
        fasta::Record::with_attrs("seq2", None, b"CGTA"),
    ];
    let temp_dir = tempdir().unwrap();
    let output_path = temp_dir.path().join("output.fasta");
    write_sequences_to_file(&sequences, output_path.clone()).unwrap();
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
    let sequences = vec![
        fasta::Record::with_attrs("seq1", None, b"ATGC"),
        fasta::Record::with_attrs("seq2", None, b"CGTA"),
    ];
    let temp_dir = tempdir().unwrap();
    let output_path = temp_dir
        .path()
        .join("nonexistent_folder")
        .join("output.fasta");
    assert!(write_sequences_to_file(&sequences, output_path).is_err());
}

#[test]
fn test_write_sequences_to_existing_file() {
    let sequences = vec![
        fasta::Record::with_attrs("seq1", None, b"ATGC"),
        fasta::Record::with_attrs("seq2", None, b"CGTA"),
    ];
    let temp_dir = tempdir().unwrap();
    let output_path = temp_dir.path().join("output.fasta");
    File::create(&output_path).unwrap();
    assert!(write_sequences_to_file(&sequences, output_path).is_err());
}

#[test]
fn test_write_newick_to_file() {
    let newick = "(((A:1.4,B:2.45):1,(D:1.2,E:2.1):1):0);";
    let trees = from_newick_string(newick).unwrap();
    let temp_dir = tempdir().unwrap();
    let output_path = temp_dir.path().join("output.newick");

    write_newick_to_file(&trees, output_path.clone()).unwrap();

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
    let mut trees = from_newick_string(newick0).unwrap();
    trees.extend(from_newick_string(newick1).unwrap());
    trees.extend(from_newick_string(newick2).unwrap());

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
    let newick = "(((A:1.4,B:2.45):1,(D:1.2,E:2.1):1):0);";
    let trees = from_newick_string(newick).unwrap();
    let temp_dir = tempdir().unwrap();
    let output_path = temp_dir
        .path()
        .join("nonexistent_folder")
        .join("output.newick");
    assert!(write_newick_to_file(&trees, output_path).is_err());
}

#[test]
fn test_write_newick_to_existing_file() {
    let newick = "(((A:1.4,B:2.45):1,(D:1.2,E:2.1):1):0);";
    let trees = from_newick_string(newick).unwrap();
    let temp_dir = tempdir().unwrap();
    let output_path = temp_dir.path().join("output.newick");
    File::create(&output_path).unwrap();
    assert!(write_newick_to_file(&trees, output_path).is_err());
}
