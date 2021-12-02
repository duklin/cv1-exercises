import cv2
import numpy as np
import maxflow

def question_3(I, rho=0.7, pairwise_cost_same=0.005, pairwise_cost_diff=0.2):

    h, w = I.shape
    num_pixels = h * w
    ### 1) Define Graph
    g = maxflow.Graph[float](num_pixels, num_pixels)

    ### 2) Add pixels as nodes
    grid_nodes = g.add_grid_nodes(I.shape)

    ### 3) Compute Unary cost

    ## Edges to source
    U0_pairwise = np.zeros_like(I)
    U0_pairwise[1:-1, 1:-1] = 2 * pairwise_cost_same    # 4 neighborhood
    U0_pairwise[0, 1:] = pairwise_cost_same             # 2/3 neighborhood
    U0_pairwise[1:, 0] = pairwise_cost_same             # 2/3 neighborhood
    U0_pairwise[1:, -1] = 2 * pairwise_cost_same        # 2/3 neighborhood
    U0_pairwise[-1, 1:-1] = 2 * pairwise_cost_same      # 2/3 neighborhood

    ## Edges to sink
    U1_pairwise = np.zeros_like(I)
    U1_pairwise[1:-1, 1:-1] = 2 * pairwise_cost_same    # 4 neighborhood
    U1_pairwise[0, 1:-1] = 2 * pairwise_cost_same       # 3 neighborhood
    U1_pairwise[1:-1, 0] = 2 * pairwise_cost_same       # 3 neighborhood
    U1_pairwise[1:-1, -1] = pairwise_cost_same          # 3 neighborhood
    U1_pairwise[-1, 1:-1] = pairwise_cost_same          # 3 neighborhood
    U1_pairwise[0, 0] = 2 * pairwise_cost_same          # 2 neighborhood
    U1_pairwise[0, -1] = pairwise_cost_same             # 2 neighborhood
    U1_pairwise[-1, 0] = pairwise_cost_same             # 2 neighborhood
    
    U0 = -np.log(np.where(I == 0, 1 - rho, rho)) + U0_pairwise
    U1 = -np.log(np.where(I == 255, 1 - rho, rho)) + U1_pairwise

    ### 4) Add terminal edges
    g.add_grid_tedges(grid_nodes, U0, U1)

    ### 5) Add Node edges
    ### Horizontal edges
    structure_hor = np.array([[0, 0, 0],
                              [pairwise_cost_diff, 0, pairwise_cost_diff - 2 * pairwise_cost_same],
                              [0, 0, 0]])
    g.add_grid_edges(grid_nodes, weights=1, structure=structure_hor)

    ### Vertical Edges
    structure_ver = structure_hor.T
    g.add_grid_edges(grid_nodes, weights=1, structure=structure_ver)
    
    ### 6) Maxflow
    g.maxflow()
    
    Denoised_I = np.where(g.get_grid_segments(grid_nodes), 255, 0)
    Denoised_I = np.array(Denoised_I, np.uint8)
    # Do not use the close button on image window to close, instead press enter (or any other key) to close windows. 
    cv2.imshow('Original Img', I)
    cv2.imshow('Denoised Img', Denoised_I), cv2.waitKey(0), cv2.destroyAllWindows()


def potts(wm, wn):
    if wm == wn:
        return 0
    return 0.35

def add_single_edge(graph, node, neighbor, node_label, neighbor_label, current_label):
    if node_label == neighbor_label == current_label:
        return
    if node_label == current_label:
        graph.add_edge(node, neighbor, 0, potts(node_label, neighbor_label))
        return
    if (node_label == neighbor_label) and (node_label != current_label):
        graph.add_edge(node, neighbor, potts(node_label, current_label), potts(current_label, node_label))
        return
    if (node_label != current_label) and (neighbor_label != current_label) and (node_label != neighbor_label):
        new_node_id = graph.add_nodes(1)
        graph.add_edge(node, new_node_id, potts(node_label, current_label), np.inf)
        graph.add_edge(new_node_id, neighbor, np.inf, potts(current_label, neighbor_label))
        graph.add_tedge(new_node_id, 0, potts(node_label, neighbor_label))
        return

def question_4(I, rho=0.6):
    labels = np.unique(I).tolist()

    Denoised_I = np.copy(I)
    ### Use Alpha expansion binary image for each label

    h, w = I.shape
    num_pixels = h * w

    Denoise_old = np.copy(I)
    while(True):
        for label in labels:
            ### 1) Define Graph
            g = maxflow.Graph[float](num_pixels, num_pixels)

            ### 2) Add pixels as nodes
            nodeid = g.add_grid_nodes(Denoised_I.shape)

            ### 3) Compute Unary costget_grid_segments
            U_source = -np.log(np.where(Denoised_I == label, rho, 0.5 * (1 - rho)))
            U_sink = np.where(Denoised_I == label, np.inf, -np.log(rho))

            ### 4) Add terminal edges
            g.add_grid_tedges(nodeid, U_source, U_sink)

            ## 5) Add Node edges
            for i in range(0, h-1):
                for j in range(0, w-1):
                    add_single_edge(g, nodeid[i, j], nodeid[i+1, j], Denoised_I[i, j], Denoised_I[i+1, j], label)
                    add_single_edge(g, nodeid[i, j], nodeid[i, j+1], Denoised_I[i, j], Denoised_I[i, j+1], label)
            for i in range(0, h-1):
                add_single_edge(g, nodeid[i, -1], nodeid[i+1, -1], Denoised_I[i, -1], Denoised_I[i+1, -1], label)

            for j in range(0, w-1):
                add_single_edge(g, nodeid[-1, j], nodeid[-1, j+1], Denoised_I[-1, j], Denoised_I[-1, j+1], label)

            # (Keep in mind the stucture of neighbourhood and set the weights according to the pairwise potential)

            ### 6) Maxflow
            g.maxflow()

            seg = g.get_grid_segments(nodeid)

            Denoised_I[seg] = label
        
        if(Denoise_old.all() == Denoised_I.all()):
            break
        Denoise_old = Denoised_I

    # Do not use the close button on image window to close, instead press enter (or any other key) to close windows. 
    cv2.imshow('Original Img', I), \
    cv2.imshow('Denoised Img', Denoised_I), cv2.waitKey(0), cv2.destroyAllWindows()


def main():
    image_q3 = cv2.imread('./images/noise.png', cv2.IMREAD_GRAYSCALE)
    image_q4 = cv2.imread('./images/noise2.png', cv2.IMREAD_GRAYSCALE)

    ### Call solution for question 3
    question_3(image_q3, rho=0.7, pairwise_cost_same=0.005, pairwise_cost_diff=0.2)
    question_3(image_q3, rho=0.7, pairwise_cost_same=0.005, pairwise_cost_diff=0.35)
    question_3(image_q3, rho=0.7, pairwise_cost_same=0.005, pairwise_cost_diff=0.55)

    ### Call solution for question 4, maybe test it 0 < rho <= 0.2 to see which one works better
    question_4(image_q4, rho=0.37)

if __name__ == "__main__":
    main()