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

    U0_pairwise = np.zeros_like(I)
    U0_pairwise[1:-1, 1:-1] = 2 * pairwise_cost_same
    U0_pairwise[0, 1:] = pairwise_cost_same
    U0_pairwise[1:, 0] = pairwise_cost_same
    U0_pairwise[1:, -1] = 2 * pairwise_cost_same
    U0_pairwise[-1, 1:-1] = 2 * pairwise_cost_same

    U1_pairwise = np.zeros_like(I)
    U1_pairwise[1:-1, 1:-1] = 2 * pairwise_cost_same
    U1_pairwise[0, 0] = 2 * pairwise_cost_same
    U1_pairwise[0, 1:-1] = 2 * pairwise_cost_same
    U1_pairwise[0, -1] = pairwise_cost_same
    U1_pairwise[1:-1, 0] = 2 * pairwise_cost_same
    U1_pairwise[-1, 0] = pairwise_cost_same
    U1_pairwise[1:-1, -1] = pairwise_cost_same
    U1_pairwise[-1, 1:-1] = pairwise_cost_same
    
    U0 = np.where(I == 0, 1 - rho, rho) + U0_pairwise
    U1 = np.where(I == 0, rho, 1 - rho) + U1_pairwise

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
    
    # (Keep in mind the stucture of neighbourhood and set the weights according to the pairwise potential)

    ### 6) Maxflow
    g.maxflow()
    
    Denoised_I = np.where(g.get_grid_segments(grid_nodes), 0, 255)
    Denoised_I = np.array(Denoised_I, np.uint8)
    # Do not use the close button on image window to close, instead press enter (or any other key) to close windows. 
    cv2.imshow('Original Img', I)
    cv2.imshow('Denoised Img', Denoised_I), cv2.waitKey(0), cv2.destroyAllWindows()


def question_4(I, rho=0.6):

    labels = np.unique(I).tolist()

    Denoised_I = np.zeros_like(I)
    ### Use Alpha expansion binary image for each label

    ### 1) Define Graph

    ### 2) Add pixels as nodes

    ### 3) Compute Unary cost

    ### 4) Add terminal edges

    ### 5) Add Node edges
    ### Vertical Edges

    ### Horizontal edges
    # (Keep in mind the stucture of neighbourhood and set the weights according to the pairwise potential)

    ### 6) Maxflow


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
    # question_4(image_q4, rho=0.2)

if __name__ == "__main__":
    main()



