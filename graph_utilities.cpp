#include "graph_utilities.h"
#include <cblas.h>
#include <cmath>
#include <stdexcept>

void build_adjacency(matrix_rep_adjacency &adj, vector<float> &embeddings, float threshold) {
    if (embeddings.size() % D != 0) {
        throw std::invalid_argument("Embedding data must contain complete vectors");
    }
    if (!std::isfinite(threshold) || threshold < -1.0f || threshold > 1.0f) {
        throw std::invalid_argument("Cosine threshold must be between -1 and 1");
    }
    const size_t cnt = embeddings.size() / D;

    adj.adjacency.assign(cnt, vector<bool>(cnt, false));
    if (cnt == 0) return;

    // build vector of embedding norms
    vector<float> norms(cnt);
    norm(embeddings, norms);

    // calculating numerator for cosine similarity
    vector<float> numerator(cnt * cnt);
    matrix_mul(embeddings, numerator, D);

    // zero-norm vectors remain isolated. Compute each undirected edge once.
    for (size_t i = 0; i < cnt; ++i) {
        if (norms[i] == 0.0f) continue;
        for (size_t j = i + 1; j < cnt; ++j) {
            if (norms[j] == 0.0f) continue;
            const double denominator = static_cast<double>(norms[i]) * norms[j];
            const double similarity = numerator[i * cnt + j] / denominator;
            const bool connected = std::isfinite(similarity) && similarity > threshold;
            adj.adjacency[i][j] = connected;
            adj.adjacency[j][i] = connected;
        }
    }
}

void norm(vector<float> &embeddings, vector<float> &norms) {
    const size_t cnt = embeddings.size() / D;
    norms.resize(cnt);
    for (size_t i = 0; i < cnt; ++i) {
        norms[i] = cblas_snrm2(D, embeddings.data() + i * D, 1);
    }
}

void matrix_mul(vector<float> &embeddings, vector<float> &result, size_t d) {
    const size_t cnt = embeddings.size() / d;

    // perform matrix multiplication using cblas
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasTrans,
        cnt,
        cnt,
        d,
        1.0f,
        embeddings.data(),
        d,
        embeddings.data(),
        d,
        0.0f,
        result.data(),
        cnt
    );
}
