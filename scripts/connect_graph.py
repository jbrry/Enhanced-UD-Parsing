import conllu
from conllu import parse_incr
from conllu import parse
#https://github.com/UniversalDependencies/tools/blob/7ee06e533347ec330e6a4f4ba31fb6f722165ae8/validate.py#L1006

file_path="fr_sequoia_pred.conllu"


def traverse_root_children(
    ids_to_heads, 
    nodes_reachable_by_root, 
    keep_searching_for_dependents,
    ):
    
    num_offspring_before = len(nodes_reachable_by_root)
    
    for token_id, head in ids_to_heads.items():
        if head in nodes_reachable_from_root:
            if token_id not in nodes_reachable_from_root:
                nodes_reachable_from_root.append(token_id)
    
    num_offspring_after = len(nodes_reachable_from_root)
    
    # if we didn't add any new children, then we have included all reachable nodes
    if num_offspring_before == num_offspring_after:
        keep_searching_for_dependents = False
    
    return nodes_reachable_by_root, keep_searching_for_dependents


with open(file_path, "r") as conllu_file:
    print("Reading UD instances from conllu dataset at: %s", file_path)
 
    for annotation in parse_incr(conllu_file):
        ids = [x["id"] for x in annotation if x["id"] != None]
        heads = [x["head"] for x in annotation]
        deprels = [x["deprel"] for x in annotation]
        
        # dictionary mapping ids to heads
        ids_to_heads = {}
        
        for conllu_id, head in zip(ids, heads):
            ids_to_heads[conllu_id] = head
            
        print("ids 2 heads", ids_to_heads)

        # store nodes reachable from root  
        nodes_reachable_from_root = []
        
        # 1) find root
        for token_id, head in ids_to_heads.items():
            if head == 0:
                root_index = token_id
                nodes_reachable_from_root.append(root_index)
        
        # 2) find root's immediate children
        for token_id, head in ids_to_heads.items():
            if head == root_index:
                nodes_reachable_from_root.append(token_id)
                
        keep_searching_for_dependents = True
        while keep_searching_for_dependents:
            nodes_reachable_from_root, keep_searching_for_dependents = traverse_root_children(
                ids_to_heads, 
                nodes_reachable_from_root, 
                keep_searching_for_dependents)
        
        print("reachable nodes", nodes_reachable_from_root)    
        
        # 3) find remaining tokens
        unreachable_nodes = []
        
        for token_id, head in ids_to_heads.items():
            # skip the root token and check if head is not reachable
            if head !=0 and head not in nodes_reachable_from_root:
                unreachable_nodes.append(token_id)
        
        print("unreachable nodes", unreachable_nodes)
        
        # for the unreachable nodes, we should try build treelets so we can add
        # an edge to the head of the treelet.
        
        treelets = {}
        
        for un in unreachable_nodes:
            uh = ids_to_heads[un]
            print(uh)
            
        print("end of sentence \n")
