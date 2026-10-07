import torch
import pandas as pd

def compute_recommendation(X: torch.Tensor, selected_songs: pd.DataFrame) -> list[dict]:
    # Compute top-10 recommendations in batches
    n = X.shape[0]
    batch_size = 256
    top_k = 10

    recommendations = []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)

            batch = X[start:end]
            scores = torch.mm(batch, X.T)

            # remove self-match
            row_idx = torch.arange(end - start, device = X.device)
            col_idx = torch.arange(start, end, device = X.device)
            scores[row_idx, col_idx] = -1

            top_indices = torch.topk(scores, k = top_k, dim = 1).indices.cpu().numpy()

            for local_idx, neighbors in enumerate(top_indices):
                current_title = selected_songs.iloc[start + local_idx]["title"]

                row = {"title" : current_title}
                for rank, neighbor_idx in enumerate(neighbors, start = 1):
                    row[f"rec_{rank}"] = selected_songs.iloc[int(neighbor_idx)]["title"]

                recommendations.append(row)
            print(f"Processed {end}/{n}")
    return recommendations