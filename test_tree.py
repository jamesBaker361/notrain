import torch
import pytest
from tree import build_tree,add_node_stats,cluster_embeddings,descend,sample_node,tree_sample,node_members,node_size


def clustered_bank(n_clusters=4,per_cluster=30,E=16,D=12,seed=0):
    # well-separated CLIP clusters, with latents that also differ per cluster
    g=torch.Generator().manual_seed(seed)
    centers=torch.nn.functional.normalize(torch.randn(n_clusters,E,generator=g),dim=-1)
    T=torch.cat([centers[c]+0.05*torch.randn(per_cluster,E,generator=g) for c in range(n_clusters)])
    T=torch.nn.functional.normalize(T,dim=-1)
    X=torch.cat([3*c+torch.randn(per_cluster,D,generator=g) for c in range(n_clusters)])
    labels=torch.arange(n_clusters).repeat_interleave(per_cluster)
    return X,T,centers,labels


@pytest.fixture(params=["4","silhouette"])
def built(request):
    X,T,centers,labels=clustered_bank()
    tree=build_tree(T,pca_dim=8,branching=request.param,k_max=6,c_build=10,n_init=2)
    return add_node_stats(tree,X,T),X,T,centers,labels


def test_structure(built):
    tree,X,T,centers,labels=built
    N=X.shape[0]
    assert sorted(tree["perm"].tolist())==list(range(N))
    for node,kids in enumerate(tree["children"]):
        if not kids:
            assert node_size(tree,node)==1
            continue
        # children partition the parent's contiguous slice, in order
        assert tree["start"][kids[0]]==tree["start"][node] and tree["end"][kids[-1]]==tree["end"][node]
        for a,b in zip(kids[:-1],kids[1:]):
            assert tree["end"][a]==tree["start"][b]
        assert all(tree["parent"][k]==node and tree["depth"][k]==tree["depth"][node]+1 for k in kids)
        if node_size(tree,node)<=10:
            assert len(kids)==node_size(tree,node)


def test_node_stats(built):
    tree,X,T,centers,labels=built
    for node in range(len(tree["children"])):
        members=node_members(tree,node)
        assert torch.allclose(tree["clip_mean"][node],T[members].mean(0),atol=1e-5)
        k=int(tree["internal_index"][node])
        if k>=0:
            assert torch.allclose(tree["vae_mean"][k],X[members].mean(0),atol=1e-4)
            assert torch.allclose(tree["vae_var"][k],X[members].var(0),atol=1e-4)


def test_silhouette_finds_the_clusters():
    X,T,centers,labels=clustered_bank()
    tree=build_tree(T,pca_dim=8,branching="silhouette",k_max=6,c_build=10,n_init=2)
    root_kids=tree["children"][0]
    assert len(root_kids)==4
    for kid in root_kids:
        assert len(set(labels[node_members(tree,kid)].tolist()))==1


def test_greedy_descent_follows_text(built):
    tree,X,T,centers,labels=built
    for c in range(4):
        node=descend(tree,centers[c],w=1e4,logit_scale=100.0,max_depth=5,c_search=20,generator=torch.Generator().manual_seed(0))
        assert node_size(tree,node)<=30
        assert set(labels[node_members(tree,node)].tolist())=={c}


def test_w0_descent_is_random(built):
    tree,X,T,centers,labels=built
    g=torch.Generator().manual_seed(0)
    nodes={descend(tree,centers[0],w=0.0,logit_scale=100.0,max_depth=1,c_search=1,generator=g) for _ in range(50)}
    assert nodes==set(tree["children"][0])


def test_stopping_rules(built):
    tree,X,T,centers,labels=built
    node=descend(tree,centers[0],w=1.0,logit_scale=100.0,max_depth=1,c_search=1)
    assert tree["depth"][node]==1
    node=descend(tree,centers[0],w=1.0,logit_scale=100.0,max_depth=99,c_search=1)
    assert node_size(tree,node)==1


def test_samplers(built):
    tree,X,T,centers,labels=built
    g=torch.Generator().manual_seed(0)
    node=tree["children"][0][0]
    members=X[node_members(tree,node)]
    for sampler in ["diagonal","lowrank","cfdm"]:
        out=sample_node(tree,X,node,5,sampler,g,cfdm_kwargs={"sigma":0.0,"steps":50,"start_t":0.2})
        assert out.shape==(5,X.shape[1]) and torch.isfinite(out).all()
    # low-rank samples stay in the affine span of the members
    out=sample_node(tree,X,node,5,"lowrank",g)
    mean=members.mean(0)
    basis=torch.linalg.svd(members-mean,full_matrices=False).Vh
    resid=(out-mean)-(out-mean)@basis.T@basis
    assert resid.abs().max()<1e-4
    # unsmoothed cfdm returns members
    out=sample_node(tree,X,node,5,"cfdm",g,cfdm_kwargs={"sigma":0.0,"steps":100,"start_t":0.2})
    assert torch.cdist(out,members).min(-1).values.max()<1e-3


def test_leaf_returns_its_image(built):
    tree,X,T,centers,labels=built
    leaf=next(n for n,k in enumerate(tree["children"]) if not k)
    for sampler in ["diagonal","lowrank","cfdm"]:
        out=sample_node(tree,X,leaf,3,sampler,cfdm_kwargs={"steps":5})
        assert torch.allclose(out,X[node_members(tree,leaf)].expand(3,-1))


def test_tree_sample(built):
    tree,X,T,centers,labels=built
    out,nodes=tree_sample(tree,X,centers[2],8,w=1e4,logit_scale=100.0,max_depth=5,c_search=20,
                          node_sampler="lowrank",generator=torch.Generator().manual_seed(0))
    assert out.shape==(8,X.shape[1]) and len(nodes)==8
    assert all(set(labels[node_members(tree,m)].tolist())=={2} for m in nodes)


def check_structure(tree,N,c_build=10):
    assert sorted(tree["perm"].tolist())==list(range(N))
    for node,kids in enumerate(tree["children"]):
        if not kids:
            assert node_size(tree,node)==1
            continue
        assert tree["start"][kids[0]]==tree["start"][node] and tree["end"][kids[-1]]==tree["end"][node]
        for a,b in zip(kids[:-1],kids[1:]):
            assert tree["end"][a]==tree["start"][b]
        assert all(tree["parent"][k]==node and tree["depth"][k]==tree["depth"][node]+1 for k in kids)
        if node_size(tree,node)<=c_build:
            assert len(kids)==node_size(tree,node)


@pytest.mark.parametrize("method,kwargs",[("dbscan",{}),("agglomerative",{"linkage":"ward"}),("agglomerative",{"linkage":"average"})])
def test_other_cluster_methods_structure_and_stats(method,kwargs):
    X,T,centers,labels=clustered_bank()
    tree=add_node_stats(build_tree(T,pca_dim=8,c_build=10,method=method,**kwargs),X,T)
    assert tree["cluster_method"]==method
    check_structure(tree,X.shape[0])
    for node in range(len(tree["children"])):
        members=node_members(tree,node)
        assert torch.allclose(tree["clip_mean"][node],T[members].mean(0),atol=1e-5)


@pytest.mark.parametrize("method",["dbscan","agglomerative"])
def test_other_cluster_methods_find_the_clusters(method):
    # every planted cluster is exactly the image set of some node
    X,T,centers,labels=clustered_bank()
    tree=build_tree(T,pca_dim=8,c_build=10,method=method)
    node_sets={frozenset(node_members(tree,n).tolist()) for n in range(len(tree["children"]))}
    for c in range(4):
        assert frozenset((labels==c).nonzero().squeeze(-1).tolist()) in node_sets


def test_agglomerative_is_binary_above_c_build():
    X,T,centers,labels=clustered_bank()
    tree=build_tree(T,pca_dim=8,c_build=10,method="agglomerative")
    for node,kids in enumerate(tree["children"]):
        if node_size(tree,node)>10:
            assert len(kids)==2


def test_deep_single_linkage_chain_builds():
    # points on a line with growing gaps: single linkage makes a chain deeper than Python's recursion limit
    N=1500
    T=torch.zeros(N,4)
    T[:,0]=torch.cumsum(torch.linspace(1,2,N),0)
    tree=build_tree(T,pca_dim=2,c_build=1,method="agglomerative",linkage="single")
    assert int(tree["depth"].max())>1000
    check_structure(tree,N,c_build=1)


def test_node_stats_from_float16_bank_and_optional():
    X,T,centers,labels=clustered_bank()
    tree=build_tree(T,pca_dim=8,branching="4",c_build=10,n_init=2)
    full=add_node_stats(dict(tree),X,T)
    half=add_node_stats(dict(tree),X.half(),T,chunk=7)
    assert torch.allclose(full["vae_mean"],half["vae_mean"],atol=2e-2) and torch.allclose(full["vae_var"],half["vae_var"],rtol=2e-2,atol=2e-2)
    skipped=add_node_stats(dict(tree),X,T,max_bytes=1)
    assert skipped["vae_mean"] is None and torch.allclose(skipped["clip_mean"],full["clip_mean"])
    node=skipped["children"][0][0]
    out=sample_node(skipped,X,node,4,"diagonal",torch.Generator().manual_seed(0))
    assert out.shape==(4,X.shape[1]) and torch.isfinite(out).all()


def test_cluster_on_captions():
    # images are mixed, but captions separate the clusters: a caption tree recovers them, descent follows the prompt
    X,T,centers,labels=clustered_bank()
    T_img=torch.nn.functional.normalize(torch.randn(T.shape,generator=torch.Generator().manual_seed(1)),dim=-1)
    bank={"X":X,"T":T_img,"C":T}
    assert torch.equal(cluster_embeddings(bank,"image"),T_img) and torch.equal(cluster_embeddings(bank,"caption"),T)
    mix=cluster_embeddings(bank,"mix",0.25)
    assert torch.allclose(mix,torch.nn.functional.normalize(0.75*T_img+0.25*T,dim=-1)) and torch.allclose(mix.norm(dim=-1),torch.ones(len(T)))
    E=cluster_embeddings(bank,"caption")
    tree=add_node_stats(build_tree(E,pca_dim=8,branching="silhouette",k_max=6,c_build=10,n_init=2),X,E)
    assert all(len(set(labels[node_members(tree,k)].tolist()))==1 for k in tree["children"][0])
    node=descend(tree,centers[1],w=1e4,logit_scale=100.0,max_depth=5,c_search=20,generator=torch.Generator().manual_seed(0))
    assert set(labels[node_members(tree,node)].tolist())=={1}
    with pytest.raises(ValueError):
        cluster_embeddings({"X":X,"T":T_img},"caption")
