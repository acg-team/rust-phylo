#[macro_export]
macro_rules! record_wo_desc {
    ($e1:expr,$e2:expr) => {
        Record::with_attrs($e1, None, $e2)
    };
}

#[macro_export]
macro_rules! record {
    ($e1:expr,$e2:expr,$e3:expr) => {
        Record::with_attrs($e1, $e2, $e3)
    };
}

#[macro_export]
macro_rules! tree {
    ($e:expr) => {
        from_newick($e).unwrap().pop().unwrap()
    };
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
