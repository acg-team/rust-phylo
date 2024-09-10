use std::path::Path;

use approx::assert_relative_eq;
use bio::io::fasta::Record;

use crate::alignment::Sequences;
use crate::evolutionary_models::{DNAModelType::*, EvoModel, ProteinModelType::*};
use crate::likelihood::PhyloCostFunction;
use crate::optimisers::{BranchOptimiser, PhyloOptimiser, TopologyOptimiser};
use crate::phylo_info::PhyloInfoBuilder;
use crate::pip_model::PIPModel;
use crate::substitution_models::{DNASubstModel, ProteinSubstModel};
use crate::tree::tree_parser::from_newick_string;

#[test]
fn k80_topo_optimisation() {
    let tree =
        from_newick_string("(((A:1.0,B:1.0)E:2.0,(C:1.0,D:1.0)F:2.0)G:3.0);").unwrap()[0].clone();
    let sequences = Sequences::new(vec![
        Record::with_attrs("A", None, b"CTATATATAC"),
        Record::with_attrs("B", None, b"ATATATATAA"),
        Record::with_attrs("C", None, b"TTATATATAT"),
        Record::with_attrs("D", None, b"TTATATATAT"),
    ]);
    let info = PhyloInfoBuilder::build_from_objects(sequences, tree).unwrap();
    let k80 = DNASubstModel::new(K80, &[4.0, 1.0]).unwrap();
    let unopt_logl = k80.cost(&info);
    let o = TopologyOptimiser::new(&k80, &info).run().unwrap();

    k80.reset();
    assert!(o.final_logl >= unopt_logl);
    assert_relative_eq!(o.final_logl, k80.cost(&o.i));
}

#[test]
fn k80_sim_topo_optimisation_from_nj() {
    let info = PhyloInfoBuilder::new(Path::new("./data/sim/K80/K80.fasta").to_path_buf())
        .build()
        .unwrap();
    let k80 = DNASubstModel::new(K80, &[4.0, 1.0]).unwrap();
    let unopt_logl = k80.cost(&info);
    let o = TopologyOptimiser::new(&k80, &info).run().unwrap();

    k80.reset();
    assert!(o.final_logl >= unopt_logl);
    assert_relative_eq!(o.final_logl, k80.cost(&o.i));
    assert_relative_eq!(o.final_logl, -4060.91963, epsilon = 1e-5);
}

#[test]
fn topology_opt_simulated_from_tree() {
    let fldr = Path::new("./data/sim/K80");
    let info = PhyloInfoBuilder::with_attrs(fldr.join("K80.fasta"), fldr.join("../tree.newick"))
        .build()
        .unwrap();
    let jc69 = DNASubstModel::new(JC69, &[]).unwrap();
    let unopt_logl = jc69.cost(&info);
    let o = TopologyOptimiser::new(&jc69, &info).run().unwrap();

    jc69.reset();
    assert!(o.final_logl >= unopt_logl);
    assert_relative_eq!(o.final_logl, jc69.cost(&o.i), epsilon = 1e-5);

    let phyml_info =
        PhyloInfoBuilder::with_attrs(fldr.join("K80.fasta"), fldr.join("phyml_tree.newick"))
            .build()
            .unwrap();

    jc69.reset();
    assert_relative_eq!(o.final_logl, jc69.cost(&phyml_info), epsilon = 1e-5);
    assert_relative_eq!(o.final_logl, -4038.721121221992, epsilon = 1e-5);

    let taxa = ["Gorilla", "Orangutan", "Gibbon", "Human", "Chimpanzee"];
    for taxon in taxa.iter() {
        assert_relative_eq!(
            o.i.tree.node(&o.i.tree.idx(taxon)).blen,
            phyml_info.tree.node(&phyml_info.tree.idx(taxon)).blen,
            epsilon = 1e-5
        );
    }
    assert_relative_eq!(o.i.tree.height, phyml_info.tree.height, epsilon = 1e-5);
    assert_eq!(o.i.tree.robinson_foulds(&phyml_info.tree), 0);
}

#[test]
fn topology_opt_simulated_from_wrong_tree() {
    let fldr = Path::new("./data/sim/K80");
    let info =
        PhyloInfoBuilder::with_attrs(fldr.join("K80.fasta"), fldr.join("../wrong_tree.newick"))
            .build()
            .unwrap();
    let jc69 = DNASubstModel::new(JC69, &[]).unwrap();
    let unopt_logl = jc69.cost(&info);
    let o = TopologyOptimiser::new(&jc69, &info).run().unwrap();

    jc69.reset();
    assert!(o.final_logl >= unopt_logl);
    assert_relative_eq!(o.final_logl, jc69.cost(&o.i), epsilon = 1e-5);

    let phyml_info =
        PhyloInfoBuilder::with_attrs(fldr.join("K80.fasta"), fldr.join("phyml_tree.newick"))
            .build()
            .unwrap();
    jc69.reset();
    assert_relative_eq!(o.final_logl, jc69.cost(&phyml_info), epsilon = 1e-5);
    assert_relative_eq!(o.final_logl, -4038.721121221992, epsilon = 1e-5);

    let taxa = ["Gorilla", "Orangutan", "Gibbon", "Chimpanzee"];
    for taxon in taxa.iter() {
        assert_relative_eq!(
            o.i.tree.by_id(taxon).blen,
            phyml_info.tree.by_id(taxon).blen,
            epsilon = 1e-5
        );
    }
    // Slightly different rooting leads to a different breakdown of branches,
    // but sum is still same
    assert_relative_eq!(
        o.i.tree.by_id("Human").blen
            + o.i
                .tree
                .node(&(o.i.tree.by_id("Chimpanzee")).parent.unwrap())
                .blen,
        phyml_info.tree.by_id("Human").blen,
        epsilon = 1e-5
    );
    assert_relative_eq!(o.i.tree.height, phyml_info.tree.height, epsilon = 1e-4);
    assert_eq!(o.i.tree.robinson_foulds(&phyml_info.tree), 0);
}

#[test]
fn protein_topology_optimisation_given_tree_start() {
    let fldr = Path::new("./data/phyml_protein_example/");
    let out_tree = fldr.join("optimisation_tree_start.newick");
    let phyml_info = PhyloInfoBuilder::with_attrs(
        fldr.join("nogap_seqs.fasta"),
        fldr.join("phyml_result.newick"),
    )
    .build()
    .unwrap();
    let wag = ProteinSubstModel::new(WAG, &[]).unwrap();
    let phyml_logl = wag.cost(&phyml_info);

    // use crate::io::write_newick_to_file;
    // let info = PhyloInfoBuilder::with_attrs(fldr.join("nogap_seqs.fasta"), fldr.join("tree.newick"))
    //     .build()
    //     .unwrap();
    // let wag = ProteinSubstModel::new(WAG, &[]).unwrap();
    // let unopt_logl = wag.cost(&info);
    // let o = TopologyOptimiser::new(&wag, &info).run().unwrap();
    // let _ = write_newick_to_file(&[o.i.tree.clone()], out_tree);
    // assert!(o.final_logl >= unopt_logl);
    // let tree = o.i.tree;
    // let logl = o.final_logl;
    // Ran in ~215.21s
    // PhyML runs in 3s

    // Optimisation can take long, tree checked is output of above
    let info = PhyloInfoBuilder::with_attrs(fldr.join("nogap_seqs.fasta"), out_tree)
        .build()
        .unwrap();
    wag.reset();
    let logl = wag.cost(&info);
    let tree = info.tree;

    // Compare tree and logl to PhyML output
    assert_relative_eq!(tree.height, phyml_info.tree.height, epsilon = 1e-3);
    assert_eq!(tree.robinson_foulds(&phyml_info.tree), 0);
    assert_relative_eq!(logl, phyml_logl, epsilon = 1e0);
}

#[test]
fn protein_topology_optimisation_nj_start() {
    let fldr = Path::new("./data/phyml_protein_example/");
    let out_tree = fldr.join("optimisation_nj_start.newick");
    let phyml_info = PhyloInfoBuilder::with_attrs(
        fldr.join("nogap_seqs.fasta"),
        fldr.join("phyml_result.newick"),
    )
    .build()
    .unwrap();
    let wag = ProteinSubstModel::new(WAG, &[]).unwrap();
    let phyml_logl = wag.cost(&phyml_info);

    // use crate::io::write_newick_to_file;
    // let info = PhyloInfoBuilder::new(fldr.join("nogap_seqs.fasta"))
    //     .build()
    //     .unwrap();
    // let wag = ProteinSubstModel::new(WAG, &[]).unwrap();
    // let unopt_logl = wag.cost(&info);
    // let o = TopologyOptimiser::new(&wag, &info).run().unwrap();
    // let _ = write_newick_to_file(&[o.i.tree.clone()], out_tree);
    // assert!(o.final_logl >= unopt_logl);
    // let tree = o.i.tree;
    // let logl = o.final_logl;
    // Ran in ~218.05s

    // Optimisation itself can take long, tree checked is output of above
    let info = PhyloInfoBuilder::with_attrs(fldr.join("nogap_seqs.fasta"), out_tree)
        .build()
        .unwrap();
    wag.reset();
    let logl = wag.cost(&info);
    let tree = info.tree;

    // Compare tree height and logl to the output of PhyML
    assert_relative_eq!(tree.height, phyml_info.tree.height, epsilon = 1e-3);
    assert_eq!(tree.robinson_foulds(&phyml_info.tree), 0);
    assert_relative_eq!(logl, phyml_logl, epsilon = 1e0);
}

#[test]
fn pip_vs_subst_dna_tree() {
    let info = &PhyloInfoBuilder::new(Path::new("./data/sim/K80/K80.fasta").to_path_buf())
        .build()
        .unwrap();
    let model = DNASubstModel::new(K80, &[4.0, 1.0]).unwrap();
    let k80_opt_res = TopologyOptimiser::new(&model, info).run().unwrap();

    let pip = PIPModel::<DNASubstModel>::new(K80, &[0.5, 0.4, 4.0, 1.0]).unwrap();
    let pip_opt_res = TopologyOptimiser::new(&pip, info).run().unwrap();

    // Tree topologies under PIP+K80 and K80 should match
    assert_eq!(pip_opt_res.i.tree.robinson_foulds(&k80_opt_res.i.tree), 0);

    // Check that likelihoods under substitution model are similar for both trees
    // but reoptimise branch lengths for PIP tree because they are not comparable
    model.reset();
    let o_pip_blen = BranchOptimiser::new(&model, &pip_opt_res.i).run().unwrap();
    assert_relative_eq!(
        k80_opt_res.final_logl,
        o_pip_blen.final_logl,
        epsilon = 1e-6
    );

    // Check that likelihoods under PIP are similar for both trees
    // but reoptimise branch lengths for substitution tree because they are not comparable
    pip.reset();
    let o_k80_blen = BranchOptimiser::new(&pip, &k80_opt_res.i).run().unwrap();
    assert_relative_eq!(
        pip_opt_res.final_logl,
        o_k80_blen.final_logl,
        epsilon = 1e-6
    );
}

#[test]
fn pip_vs_subst_protein_tree_nogaps() {
    let fldr = Path::new("./data/phyml_protein_example/");
    let out_tree = fldr.join("optimisation_pip_nj_start.newick");
    // use crate::io::write_newick_to_file;
    // let info = PhyloInfoBuilder::new(fldr.join("nogap_seqs.fasta"))
    //     .build()
    //     .unwrap();
    // let pip = PIPModel::<ProteinSubstModel>::new(WAG, &[50.0, 0.1]).unwrap();
    // let unopt_logl = pip.cost(&info);
    // let o = TopologyOptimiser::new(&pip, &info).run().unwrap();
    // let _ = write_newick_to_file(&[o.i.tree.clone()], out_tree);
    // assert!(o.final_logl >= unopt_logl);
    // let pip_opt_info = o.i;
    // Ran in ~830.78s

    // Optimisation takes too long, tree checked is output of above
    let pip_opt_info = PhyloInfoBuilder::with_attrs(fldr.join("nogap_seqs.fasta"), out_tree)
        .build()
        .unwrap();

    let subst_opt_info = PhyloInfoBuilder::with_attrs(
        fldr.join("nogap_seqs.fasta"),
        fldr.join("optimisation_nj_start.newick"),
    )
    .build()
    .unwrap();

    // compare tree created with a substitution model to the one with PIP
    assert_eq!(pip_opt_info.tree.robinson_foulds(&subst_opt_info.tree), 0);

    // Check that likelihoods under same model are similar for both trees
    let wag = ProteinSubstModel::new(WAG, &[]).unwrap();
    let pip_reopt_result = BranchOptimiser::new(&wag, &pip_opt_info).run().unwrap();
    wag.reset();
    assert_relative_eq!(
        wag.cost(&subst_opt_info),
        pip_reopt_result.final_logl,
        epsilon = 1e-5
    );

    // Check that the likelihoods under the same model are similar for both trees
    let pip_wag = PIPModel::<ProteinSubstModel>::new(WAG, &[50.0, 0.1]).unwrap();
    let o2_subst = BranchOptimiser::new(&pip_wag, &subst_opt_info)
        .run()
        .unwrap();
    pip_wag.reset();
    assert_relative_eq!(
        pip_wag.cost(&pip_opt_info),
        o2_subst.final_logl,
        epsilon = 1e-5
    );
}

#[test]
fn protein_optimise_model_tree() {
    // let info = PhyloInfoBuilder::new(PathBuf::from("./data/phyml_protein_example/seqs.fasta"))
    //     .build()
    //     .unwrap();
    // let model = PIPModel::<ProteinSubstModel>::new(WAG, &[1.4, 0.5]).unwrap();
    // let unopt_logl = model.cost(&info);
    // let o = ModelOptimiser::new(&model, &info, Empirical).run().unwrap();
    // let model_opt_logl = o.final_logl;
    // assert!(model_opt_logl >= unopt_logl);
    // let o = TopologyOptimiser::new(&o.model, &info).run().unwrap();
    // assert!(model_opt_logl >= unopt_logl);
    // assert!(o.final_logl >= unopt_logl);
    // // The above ran in 703.87s

    // let info = PhyloInfoBuilder::new(PathBuf::from("./data/phyml_protein_example/seqs.fasta"))
    //     .build()
    //     .unwrap();
    // let model = PIPModel::<ProteinSubstModel>::new(WAG, &[1.4, 0.5]).unwrap();
    // let unopt_logl = model.cost(&info);
    // let o = TopologyOptimiser::new(&model, &info).run().unwrap();
    // assert!(o.final_logl >= unopt_logl);
    // The above ran in 497.22s

    let optim_model = PIPModel::<ProteinSubstModel>::new(WAG, &[49.56941, 0.09352]).unwrap();
    let with_model_optim = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/phyml_protein_example/seqs.fasta"),
        PathBuf::from(
            "./data/phyml_protein_example/optimisation_pip_nj_start_model_optim_gaps.newick",
        ),
    )
    .build()
    .unwrap();

    let model = ProteinSubstModel::new(WAG, &[1.4, 0.5]).unwrap();
    let wo_model_optim = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/phyml_protein_example/seqs.fasta"),
        PathBuf::from(
            "./data/phyml_protein_example/optimisation_pip_nj_start_model_optim_gaps.newick",
        ),
    )
    .build()
    .unwrap();

    assert!(with_model_optim.tree.robinson_foulds(&wo_model_optim.tree) == 0);
    assert!(optim_model.cost(&with_model_optim) > model.cost(&wo_model_optim));
}
