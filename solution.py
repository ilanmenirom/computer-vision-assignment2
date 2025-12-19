"""Stereo matching."""
import numpy as np
from scipy.signal import convolve2d


class Solution:
    def __init__(self):
        pass

    @staticmethod
    def ssd_distance(left_image: np.ndarray,
                     right_image: np.ndarray,
                     win_size: int,
                     dsp_range: int) -> np.ndarray:
        """Compute the SSDD distances tensor.

        Args:
            left_image: Left image of shape: HxWx3, and type np.double64.
            right_image: Right image of shape: HxWx3, and type np.double64.
            win_size: Window size odd integer.
            dsp_range: Half of the disparity range. The actual range is
            -dsp_range, -dsp_range + 1, ..., 0, 1, ..., dsp_range.

        Returns:
            A tensor of the sum of squared differences for every pixel in a
            window of size win_size X win_size, for the 2*dsp_range + 1
            possible disparity values. The tensor shape should be:
            HxWx(2*dsp_range+1).
        """
        num_of_rows, num_of_cols = left_image.shape[0], left_image.shape[1]
        disparity_values = range(-dsp_range, dsp_range + 1)
        ssdd_tensor = np.zeros((num_of_rows,
                                num_of_cols,
                                len(disparity_values)))
        """INSERT YOUR CODE HERE"""

        # Calculate padding and pad both images (only for window calculations:
        pad = int(np.floor(win_size / 2))
        padded_left = Solution.pad_image(left_image, pad)
        padded_right = Solution.pad_image(right_image, pad)

        # Scan the left image row by row:
        for row in range(num_of_rows):
            for col in range(num_of_cols):
                # Fix the current left-image window
                left_win = padded_left[row:row + win_size, col:col + win_size]
                # Scan +- disparity range windows in right image, if they exist:
                for val in disparity_values:
                    running_col = col + val
                    dsp_idx = val + dsp_range
                    if 0 <= running_col < num_of_cols:
                        right_win = padded_right[row:row + win_size, running_col:running_col + win_size]
                        diff = left_win - right_win
                        ssd = np.sum(diff * diff)
                    else:
                        ssd = np.sum(left_win * left_win)

                    ssdd_tensor[row, col, dsp_idx] = ssd

            print('row:', row, 'of:', num_of_rows)

        # Normalize:
        ssdd_tensor -= ssdd_tensor.min()
        ssdd_tensor /= ssdd_tensor.max()
        ssdd_tensor *= 255.0

        return ssdd_tensor

    @staticmethod
    def naive_labeling(ssdd_tensor: np.ndarray) -> np.ndarray:
        """Estimate a naive depth estimation from the SSDD tensor.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.

        Evaluate the labels in a naive approach. Each value in the
        result tensor should contain the disparity matching minimal ssd (sum of
        squared difference).

        Returns:
            Naive labels HxW matrix.
        """
        # you can erase the label_no_smooth initialization.
        label_no_smooth = np.zeros((ssdd_tensor.shape[0], ssdd_tensor.shape[1]))
        """INSERT YOUR CODE HERE"""
        label_no_smooth = np.argmin(ssdd_tensor, axis=2)
        return label_no_smooth

    @staticmethod
    def dp_grade_slice(c_slice: np.ndarray, p1: float, p2: float) -> np.ndarray:
        """Calculate the scores matrix for slice c_slice.

        Calculate the scores slice which for each column and disparity value
        states the score of the best route. The scores slice is of shape:
        (2*dsp_range + 1)xW.

        Args:
            c_slice: A slice of the ssdd tensor.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Scores slice which for each column and disparity value states the
            score of the best route.
        """
        num_labels, num_of_cols = c_slice.shape[0], c_slice.shape[1]
        l_slice = np.zeros((num_labels, num_of_cols))
        """INSERT YOUR CODE HERE"""
        # Initialize first row with raw values"
        l_slice[:, 0] = c_slice[:, 0]
        pad_object = 1e9 * np.ones(shape=(1, num_of_cols))
        l_padded = np.concatenate([pad_object, c_slice, pad_object], axis=0)

        # Storing items in memory to avoid computations within loop:
        l_penalized = l_padded + p2
        penalty_diff = np.array([p1 - p2, -p2, p1 - p2])

        for col in range(1, num_of_cols):
            l_padded[:, col] -= l_padded[:, col - 1].min()
            l_penalized[:, col - 1] = l_padded[:, col - 1] + p2
            mdc = np.zeros(l_padded.shape[0])
            for label in range(1, num_labels + 1):
                vec = l_penalized[:, col - 1].copy()
                vec[label - 1:label + 2] += penalty_diff  # Subtract penalties from penalized array
                mdc[label] = vec.min()

            l_padded[:, col] += mdc

        l_slice = l_padded[1:-1, :]

        return l_slice

    def dp_labeling(self,
                    ssdd_tensor: np.ndarray,
                    p1: float,
                    p2: float) -> np.ndarray:
        """Estimate a depth map using Dynamic Programming.

        (1) Call dp_grade_slice on each row slice of the ssdd tensor.
        (2) Store each slice in a corresponding l tensor (of shape as ssdd).
        (3) Finally, for each pixel in l (along each row and column), choose
        the best disparity value. That is the disparity value which
        corresponds to the lowest l value in that pixel.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Dynamic Programming depth estimation matrix of shape HxW.
        """
        l = np.zeros_like(ssdd_tensor)
        """INSERT YOUR CODE HERE"""

        num_rows, num_cols, num_disparities = ssdd_tensor.shape
        # mark -1 as irrelevant, to handle manipulations from diagonal paths:
        final_labels = np.full((num_rows, num_cols), -1, dtype=int)
        for row in range(num_rows):
            ssdd_row = ssdd_tensor[row, :, :]
            relevant_cols = np.where(~np.isnan(ssdd_row[:,0]))[0] # Handle nan, for shorter slices of diagonal paths
            row_slice = ssdd_row[relevant_cols, :].transpose()
            accumulated_cost_slice = self.dp_grade_slice(row_slice, p1, p2)
            final_labels[row, relevant_cols] = self._backtrack_slice(accumulated_cost_slice, p1, p2)

        return final_labels  # includes -1 for irrelevant areas (caused from diagonal paths)

    @staticmethod
    def _backtrack_slice(accumulated_cost: np.ndarray, p1: float, p2: float) -> np.ndarray:
        """
        Performs the backward pass (backtracking) to find the optimal path.

        Args:
            accumulated_cost: The cost matrix (D x W) computed by the forward pass.
                              D = number of disparities, W = width of image.
            p1: Penalty for disparity jump of 1.
            p2: Penalty for disparity jump > 1.

        Returns:
            A 1D array of size W containing the optimal labels (indices) for this row.
        """
        num_labels, num_cols = accumulated_cost.shape
        labels = np.zeros(num_cols, dtype=int)

        # 1. Start from the last column: simply pick the global minimum
        # This determines the anchor point for the backward path.
        labels[-1] = np.argmin(accumulated_cost[:, -1])

        # 2. Backtrack from W-2 down to 0
        for col in range(num_cols - 2, -1, -1):
            next_label = labels[col + 1]  # The label we chose for the pixel to the right
            transition_costs = np.full(num_labels, p2)

            # Zero cost if we stay the same
            transition_costs[next_label] = 0.0

            # p1 cost if we move by 1
            if next_label > 0:
                transition_costs[next_label - 1] = p1
            if next_label < num_labels - 1:
                transition_costs[next_label + 1] = p1

            # Total cost = Cost at this node + Cost to transition to the next node
            total_cost_at_step = accumulated_cost[:, col] + transition_costs

            # Choose the disparity that minimizes this combined cost
            labels[col] = np.argmin(total_cost_at_step)

        return labels

    def dp_labeling_per_direction(self,
                                  ssdd_tensor: np.ndarray,
                                  p1: float,
                                  p2: float) -> dict:
        """Return a dictionary of directions to a Dynamic Programming
        etimation of depth.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Dictionary int->np.ndarray which maps each direction to the
            corresponding dynamic programming estimation of depth based on
            that direction.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        direction_to_slice = {}
        """INSERT YOUR CODE HERE"""
        num_rows, num_cols, num_disparities = ssdd_tensor.shape

        # Horizontally:
        direction_to_slice[1] = self.dp_labeling(ssdd_tensor, p1, p2)
        direction_to_slice[5] = self._dp_labeling_opposite(ssdd_tensor, p1, p2)

        # Vertically:
        # rotate to apply DP on columns:
        ssdd_tensor_rotated = np.rot90(ssdd_tensor)
        vertical_labels_3 = self.dp_labeling(ssdd_tensor_rotated, p1, p2)
        vertical_labels_7 = self._dp_labeling_opposite(ssdd_tensor_rotated, p1, p2)
        # rotate back to original dimensions
        direction_to_slice[3] = np.rot90(vertical_labels_3, 3)
        direction_to_slice[7] = np.rot90(vertical_labels_7, 3)

        # Diagonals 2 and 6:
        ssdd_tensor_diag2 = self.manipulate_ssdd_diagonally(ssdd_tensor)
        diag2_labels = self.dp_labeling(ssdd_tensor_diag2, p1, p2)
        diag6_labels = self._dp_labeling_opposite(ssdd_tensor_diag2, p1, p2)
        direction_to_slice[2] = self.inverse_diagonal(diag2_labels, num_rows, num_cols)
        direction_to_slice[6] = self.inverse_diagonal(diag6_labels, num_rows, num_cols)

        # Diagonals 4 and 8:
        # flip SSD tensor, to achieve diagonals from other side
        ssdd_tensor_diag4 = self.manipulate_ssdd_diagonally(ssdd_tensor[:, ::-1, :])
        diag4_labels = self.dp_labeling(ssdd_tensor_diag4, p1, p2)
        diag8_labels = self._dp_labeling_opposite(ssdd_tensor_diag4, p1, p2)
        # flip back to adapt to the original image indices
        direction_to_slice[4] = self.inverse_diagonal(diag4_labels, num_rows, num_cols)[:, ::-1]
        direction_to_slice[8] = self.inverse_diagonal(diag8_labels, num_rows, num_cols)[:, ::-1]

        return direction_to_slice

    def _dp_labeling_opposite(self,
                              ssdd_tensor: np.ndarray,
                              p1: float,
                              p2: float) -> np.ndarray:
        # flip x before labeling and after, to receive slices in the opposite directions in the same indices
        return self.dp_labeling(ssdd_tensor[:, ::-1, :], p1, p2)[:, ::-1]


    def sgm_labeling(self, ssdd_tensor: np.ndarray, p1: float, p2: float):
        """Estimate the depth map according to the SGM algorithm.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Semi-Global Mapping depth estimation matrix of shape HxW.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        """INSERT YOUR CODE HERE"""
        return self.naive_labeling(l)

    @staticmethod
    def pad_image(img, pad_add):
        if len(img.shape) > 2:
            pad_size = [img.shape[0] + 2 * pad_add, img.shape[1] + 2 * pad_add, img.shape[2]]
        else:
            pad_size = [img.shape[0] + 2 * pad_add, img.shape[1] + 2 * pad_add]

        padded_img = np.zeros(shape=pad_size)
        padded_img[pad_add:(img.shape[0] + pad_add), pad_add:(img.shape[1] + pad_add)] = img
        return padded_img

    @staticmethod
    def manipulate_ssdd_diagonally(ssdd_tensor: np.ndarray) -> np.ndarray:
        num_rows, num_cols, num_disparities = ssdd_tensor.shape
        num_diagonals = num_rows + num_cols - 1
        max_diagonals_len = min(num_rows, num_cols)
        # Build rows as diagonals:
        ssdd_tensor_diagonal = np.full([num_diagonals, max_diagonals_len, num_disparities], np.nan)
        offsets_arr = np.arange(-(num_rows - 1), num_cols)
        for offset_ind, offset in enumerate(offsets_arr):
            diag = ssdd_tensor.diagonal(offset).transpose()  # shape: diagonal axis, disparities
            ssdd_tensor_diagonal[offset_ind, :diag.shape[0], :] = diag

        return ssdd_tensor_diagonal

    @staticmethod
    def inverse_diagonal(diag_mat: np.ndarray, num_rows: int, num_cols: int) -> np.ndarray:
        assert diag_mat.shape[0] == num_rows + num_cols - 1
        assert diag_mat.shape[1] == min(num_rows, num_cols)
        offsets_arr = np.arange(-(num_rows - 1), num_cols)
        res = np.full([num_rows, num_cols], np.nan)
        jump_size = num_cols + 1  # to advance in row and column through linear indexing
        mat_numel = num_rows * num_cols

        for offset_ind, offset in enumerate(offsets_arr):
            # The first elements in the diagonal is on upper side of the matrix when offset>=0, and on the left
            # side of the matrix when offset < 0:
            starting_linear_ind = offset if offset >= 0 else -offset * num_cols

            if offset >= 0:
                num_pixels = min(num_rows, num_cols - offset)
            else:
                num_pixels = min(num_rows + offset, num_cols)

            linear_inds = np.arange(starting_linear_ind, mat_numel, jump_size)

            linear_inds = linear_inds[:num_pixels]

            r_inds, c_inds = np.unravel_index(linear_inds, (num_rows, num_cols))
            res[r_inds, c_inds] = diag_mat[offset_ind, :len(linear_inds)]

        return res