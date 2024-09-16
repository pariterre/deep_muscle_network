import torch


def get_pareto_front_indices(x_data: torch.Tensor, y_data: torch.Tensor) -> set[int]:
    """
    Find the indices of the points that are on the Pareto front.

    Parameters
    ----------
    x_data : torch.Tensor
        The x values.
    y_data : torch.Tensor
        The y values.

    Returns
    -------
    set[int]
        The indices of the points that are on the Pareto front.
    """

    # Check if the x and y data have the same length
    if len(x_data) != len(y_data):
        raise ValueError("The x and y data must have the same length.")

    point_count = len(x_data)

    # Find the Pareto front points
    pareto_indices = set()
    for i in range(point_count):
        for j in range(point_count):
            if i == j:
                continue

            # Check if the point i is dominated by the point j
            if x_data[j] <= x_data[i] and y_data[j] <= y_data[i]:
                break
        else:
            pareto_indices.add(i)

    return pareto_indices
