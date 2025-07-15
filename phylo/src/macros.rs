#[macro_export]
macro_rules! record_wo_desc {
    ($e1:expr,$e2:expr) => {{
        use bio::io::fasta::Record;
        Record::with_attrs($e1, None, $e2)
    }};
}

#[macro_export]
macro_rules! record {
    ($e1:expr,$e2:expr,$e3:expr) => {{
        use bio::io::fasta::Record;
        Record::with_attrs($e1, $e2, $e3)
    }};
}

#[macro_export]
macro_rules! tree {
    ($e:expr) => {{
        use $crate::tree::tree_parser::from_newick;
        from_newick($e).unwrap().pop().unwrap()
    }};
}

#[macro_export]
macro_rules! align {
    ($e:expr) => {{
        use $crate::alphabets::GAP;
        let mut i = 0;
        $e.iter()
            .map(|&byte| {
                if byte == GAP {
                    None
                } else {
                    i += 1;
                    Some(i - 1)
                }
            })
            .collect::<Vec<_>>()
    }};
}

#[macro_export]
macro_rules! site {
    ($s:expr, $f:expr) => {{
        use $crate::alphabets::ParsimonySet;
        use $crate::parsimony::ParsimonySite;
        ParsimonySite::new(ParsimonySet::from_slice($s), $f)
    }};
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
pub mod tests {
    use bio::io::fasta::Record;

    use crate::parsimony::{ParsimonySite, SiteFlag};
    use crate::tree::Tree;

    #[test]
    fn test_record_macro() {
        // Test basic functionality
        let record = record!("seq1", Some("description"), b"ATCG");
        assert_eq!(record.id(), "seq1");
        assert_eq!(record.desc(), Some("description"));
        assert_eq!(record.seq(), b"ATCG");

        // Test with None description
        let record = record!("seq2", None, b"TTTT");
        assert_eq!(record.id(), "seq2");
        assert_eq!(record.desc(), None);
        assert_eq!(record.seq(), b"TTTT");

        // Test with variables
        let id = "var_seq";
        let desc = Some("variable description");
        let seq = b"GCGC";
        let record = record!(id, desc, seq);
        assert_eq!(record.id(), id);
        assert_eq!(record.desc(), desc);
        assert_eq!(record.seq(), seq);
    }

    #[test]
    fn test_record_wo_desc_macro() {
        // Test basic functionality
        let record = record_wo_desc!("seq1", b"ATCG");
        assert_eq!(record.id(), "seq1");
        assert_eq!(record.seq(), b"ATCG");
        assert_eq!(record.desc(), None);

        // Test with variables
        let id = "test_seq";
        let seq = b"AAAA";
        let record = record_wo_desc!(id, seq);
        assert_eq!(record.id(), id);
        assert_eq!(record.seq(), seq);
    }

    #[test]
    fn test_tree_macro() {
        // Test simple tree
        let tree = tree!("(A,B);");
        assert_eq!(tree.leaves().len(), 2);

        // Test more complex tree
        let tree = tree!("((A,B),(C,D));");
        assert_eq!(tree.leaves().len(), 4);
        assert_eq!(tree.len(), 7); // 4 leaves + 3 internal nodes

        // Test with variable
        let newick_str = "(seq1:0.1,seq2:0.2);";
        let tree = tree!(newick_str);
        assert_eq!(tree.leaves().len(), 2);
    }

    #[test]
    fn test_align_macro() {
        // Test basic alignment
        let alignment = align!(b"01-2");
        assert_eq!(alignment, vec![Some(0), Some(1), None, Some(2)]);

        // Test all gaps
        let alignment = align!(b"---");
        assert_eq!(alignment, vec![None, None, None]);

        // Test no gaps
        let alignment = align!(b"012");
        assert_eq!(alignment, vec![Some(0), Some(1), Some(2)]);

        // Test with variable
        let input = b"0-1";
        let alignment = align!(input);
        assert_eq!(alignment, vec![Some(0), None, Some(1)]);
    }

    #[test]
    fn test_site_macro() {
        // Test that the macro compiles and creates sites correctly
        let site1 = site!(b"ATCG", SiteFlag::NoGap);
        let site2 = site!(b"AT", SiteFlag::GapFixed);
        let site3 = site!(b"A", SiteFlag::GapOpen);

        // Test that sites are not equal (they should have different content)
        assert_ne!(site1, site2);
        assert_ne!(site2, site3);

        // Test with variable
        let flag = SiteFlag::GapExt;
        let site = site!(b"GCGC", flag);

        // Just test that it compiles and creates a site
        // We can't easily test the internal state since fields are private
        let _ = site;
    }

    // Test macros work in different contexts
    mod nested_module {
        use crate::parsimony::SiteFlag;

        #[test]
        fn test_macros_in_nested_module() {
            // Test that macros work when used in nested modules
            let record = record!("nested", None, b"ATCG");
            assert_eq!(record.id(), "nested");

            let record = record_wo_desc!("nested", b"ATCG");
            assert_eq!(record.id(), "nested");

            let tree = tree!("(A,B);");
            assert_eq!(tree.leaves().len(), 2);

            let alignment = align!(b"01-2");
            assert_eq!(alignment, vec![Some(0), Some(1), None, Some(2)]);

            let site = site!(b"ACGT", SiteFlag::NoGap);
            let _ = site; // Ensure it compiles
        }
    }

    #[test]
    fn test_macros_with_complex_expressions() {
        use crate::alphabets::GAP;

        // Test macros work with complex expressions as arguments
        let id = format!("seq_{}", 1);
        let record = record_wo_desc!(id.as_str(), b"ATCG");
        assert_eq!(record.id(), "seq_1");

        // Test with method calls
        let seq_vec = vec![b"ATCG"[0], b"ATCG"[1]];
        let record = record_wo_desc!("test", seq_vec.as_slice());
        assert_eq!(record.seq(), b"AT");

        // Test tree macro with string operations
        let base = "(A,B)";
        let full_tree = format!("{base};");
        let tree = tree!(full_tree.as_str());
        assert_eq!(tree.leaves().len(), 2);

        // Test alignment macro with a u8 array
        let input = [0u8, GAP, 1, 2, GAP, 3];
        let alignment = align!(input);
        assert_eq!(
            alignment,
            vec![Some(0), None, Some(1), Some(2), None, Some(3)]
        );

        // Test site macro with a variable
        let site_content = b"ACGT";
        let site = site!(site_content, SiteFlag::NoGap);
        let _ = site; // Ensure it compiles
    }

    #[test]
    fn test_macros_in_functions() {
        fn create_record(id: &str, desc: Option<&str>, seq: &[u8]) -> Record {
            record!(id, desc, seq)
        }

        let record = create_record("func_test", Some("func_desc"), b"GGGG");
        assert_eq!(record.id(), "func_test");
        assert_eq!(record.desc(), Some("func_desc"));
        assert_eq!(record.seq(), b"GGGG");

        fn create_record_wo_desc(id: &str, seq: &[u8]) -> Record {
            record_wo_desc!(id, seq)
        }

        let record = create_record_wo_desc("func_test", b"GGGG");
        assert_eq!(record.id(), "func_test");
        assert_eq!(record.desc(), None);
        assert_eq!(record.seq(), b"GGGG");

        fn create_tree(newick: &str) -> Tree {
            tree!(newick)
        }

        let tree = create_tree("(X,Y);");
        assert_eq!(tree.leaves().len(), 2);

        fn create_alignment(input: &[u8]) -> Vec<Option<usize>> {
            align!(input)
        }

        let alignment = create_alignment(b"01-2");
        assert_eq!(alignment, vec![Some(0), Some(1), None, Some(2)]);

        fn create_site(s: &[u8], f: SiteFlag) -> ParsimonySite {
            site!(s, f)
        }

        let site = create_site(b"A", SiteFlag::NoGap);
        let _ = site; // Ensure it compiles
    }

    #[test]
    fn test_macros_in_closures() {
        let records: Vec<_> = ["seq1", "seq2"]
            .iter()
            .map(|&id| record_wo_desc!(id, b"AAAA"))
            .collect();

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].id(), "seq1");
        assert_eq!(records[1].id(), "seq2");

        let trees: Vec<_> = ["(A,B);", "(C,D);"]
            .iter()
            .map(|&newick| tree!(newick))
            .collect();

        assert_eq!(trees.len(), 2);
        assert_eq!(trees[0].leaves().len(), 2);
        assert_eq!(trees[1].leaves().len(), 2);

        // Test align! macro in closures
        let sequences = [b"01-2", b"0--1", b"012-"];
        let alignments: Vec<_> = sequences.iter().map(|&seq| align!(seq)).collect();

        assert_eq!(alignments.len(), 3);
        assert_eq!(alignments[0], vec![Some(0), Some(1), None, Some(2)]);
        assert_eq!(alignments[1], vec![Some(0), None, None, Some(1)]);
        assert_eq!(alignments[2], vec![Some(0), Some(1), Some(2), None]);

        // Test site! macro in closures
        let site_data = [
            (b"ATCG" as &[u8], SiteFlag::NoGap),
            (b"A", SiteFlag::GapFixed),
            (b"AG", SiteFlag::GapOpen),
        ];
        let sites: Vec<_> = site_data
            .iter()
            .map(|&(seq, flag)| site!(seq, flag))
            .collect();

        assert_eq!(sites.len(), 3);
        // Sites should be different from each other
        assert_eq!(sites[0], sites[0]);
        assert_ne!(sites[0], sites[1]);
        assert_ne!(sites[1], sites[2]);
        assert_ne!(sites[0], sites[2]);
    }

    #[test]
    #[allow(non_snake_case)] // Intentionally shadowing type names for hygiene testing
    fn test_macro_hygiene() {
        // Test that macros don't interfere with local variables
        let Record = "not_a_record"; // Shadow the Record type
        let _record = record_wo_desc!("test", b"ATCG"); // Should still work
        let _record = record!("test", Some("test_desc"), b"ATCG"); // Should still work
        assert_eq!(Record, "not_a_record"); // Local variable unchanged

        // Test with function names
        fn from_newick() -> &'static str {
            "not_a_function"
        }
        let _tree = tree!("(A,B);"); // Should still work
        assert_eq!(from_newick(), "not_a_function"); // Local function unchanged

        // Test with align macro - verify it uses the correct GAP constant
        mod alphabets {
            pub const GAP: u8 = 99; // Different GAP constant in local scope
        }
        let GAP = 100; // Local GAP constant
        let _alignment = align!(b"01-2"); // Should use $crate::alphabets::GAP, not local one
        assert_eq!(alphabets::GAP, 99); // Local mod GAP is different
        assert_eq!(GAP, 100); // Local GAP is unchanged

        // Test with site macro - shadow the actual types used in the macro
        let ParsimonySite = "not_a_site"; // Shadow the ParsimonySite type
        let ParsimonySet = "not_a_set"; // Shadow the ParsimonySet type
        let _site = site!(b"ATCG", SiteFlag::NoGap); // Should still work
        assert_eq!(ParsimonySite, "not_a_site"); // Local variable unchanged
        assert_eq!(ParsimonySet, "not_a_set"); // Local variable unchanged
    }
}
