def non_graph_weight_addition(graph_indices,r_weights,boundary_weight_dict,labels,alpha,beta,width,height):
    for i in graph_indices:
                
        left = i-1
        right = i+1
        up = i-width
        down = i+width
        
        size = width*height
        if left > -1:
            if i % width != 0:
                if labels[left] != alpha and labels[left] != beta:
                    try:
                        r_weights[i][alpha] += boundary_weight_dict[(i,left)]
                        r_weights[i][beta] += boundary_weight_dict[(i,left)]
                    except KeyError:
                        r_weights[i][alpha] += boundary_weight_dict[(left,i)]
                        r_weights[i][beta] += boundary_weight_dict[(left,i)]

        if right < size:
            if right % width != 0:
                if labels[right] != alpha and labels[right] != beta:
                    try:
                        r_weights[i][alpha] += boundary_weight_dict[(i,right)]
                        r_weights[i][beta] += boundary_weight_dict[(i,right)]
                    except KeyError:
                        r_weights[i][alpha] += boundary_weight_dict[(right,i)]
                        r_weights[i][beta] += boundary_weight_dict[(right,i)]

        if up > -1:        
            if labels[up] != alpha and labels[up] != beta:
                try:
                    r_weights[i][alpha] += boundary_weight_dict[(i,up)]
                    r_weights[i][beta] += boundary_weight_dict[(i,up)]
                except KeyError:
                    r_weights[i][alpha] += boundary_weight_dict[(up,i)]
                    r_weights[i][beta] += boundary_weight_dict[(up,i)]

        if down < size:        
            if labels[down] != alpha and labels[down] != beta:
                try:
                    r_weights[i][alpha] += boundary_weight_dict[(i,down)]
                    r_weights[i][beta] += boundary_weight_dict[(i,down)]
                except KeyError:
                    r_weights[i][alpha] += boundary_weight_dict[(down,i)]
                    r_weights[i][beta] += boundary_weight_dict[(down,i)]