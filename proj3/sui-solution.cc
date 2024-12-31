#include "memusage.h"
#include "search-interface.h"
#include "search-strategies.h"
#include <functional>
#include <memory>
#include <optional>
#include <queue>
#include <set>
#include <stack>
#include <vector>

const size_t mem_limit_reserve = 50'000'000;

std::vector<SearchAction> BreadthFirstSearch::solve(const SearchState& init_state) {
    std::queue<std::pair<SearchState, std::vector<SearchAction>>> queue;
    std::set<SearchState> visited;

    queue.push({init_state, {}});

    while (!queue.empty()) {
        if (getCurrentRSS() > mem_limit_ - mem_limit_reserve) {
            return {};
        }
        auto [actual_state, actual_path] = queue.front();
        queue.pop();

        if (actual_state.isFinal()) {
            return actual_path;
        }

        if (visited.find(actual_state) == visited.end()) {
            visited.insert(actual_state);

            for (auto& action : actual_state.actions()) {
                SearchState new_state = action.execute(actual_state);
                if (visited.find(new_state) == visited.end()) {
                    auto new_path = actual_path;
                    new_path.push_back(action);
                    queue.push({new_state, new_path});
                }
            }
        }
    }
    return {};
}

std::vector<SearchAction> DepthFirstSearch::solve(const SearchState& init_state) {
    std::stack<std::pair<SearchState, std::vector<SearchAction>>> stack;
    std::set<SearchState> visited;

    stack.push({init_state, {}});

    while (!stack.empty()) {
        if (getCurrentRSS() > mem_limit_ - mem_limit_reserve) {
            return {};
        }
        auto [actual_state, actual_path] = stack.top();
        stack.pop();

        if (actual_state.isFinal()) {
            return actual_path;
        }

        if (actual_path.end() - actual_path.begin() >= depth_limit_)
            continue;

        if (visited.find(actual_state) == visited.end()) {
            visited.insert(actual_state);

            for (auto& action : actual_state.actions()) {
                SearchState new_state = action.execute(actual_state);
                if (visited.find(new_state) == visited.end()) {
                    auto new_path = actual_path;
                    new_path.push_back(action);
                    stack.push({new_state, new_path});
                }
            }
        }
    }
    return {};
}

double StudentHeuristic::distanceLowerBound(const GameState& state) const { return 0; }

struct AStarNode {
    double cost;
    SearchState state;
    std::optional<SearchAction> action;
    std::shared_ptr<AStarNode> parent;

    bool operator>(const AStarNode& that) const { return this->cost > that.cost; }

    std::vector<SearchAction> path() {
        if (!action.has_value())
            return {};

        std::vector<SearchAction> path = {action.value()};
        std::shared_ptr<AStarNode> tmp = parent;
        while (tmp != nullptr) {
            if (tmp->action.has_value())
                path.insert(path.begin(), tmp->action.value());
            tmp = tmp->parent;
        }

        return path;
    }
};

std::vector<SearchAction> AStarSearch::solve(const SearchState& init_state) {
    std::set<SearchState> visited;
    std::priority_queue<AStarNode, std::vector<AStarNode>, std::greater<>> queue;

    AStarNode init = {0.0, init_state, std::nullopt, nullptr};
    queue.push(init);

    while (!queue.empty()) {
        auto current = queue.top();
        queue.pop();

        if (current.state.isFinal()) {
            return current.path();
        }

        if (visited.find(current.state) == visited.end()) {
            visited.insert(current.state);

            for (auto& action : current.state.actions()) {
                if (getCurrentRSS() > mem_limit_ - mem_limit_reserve) {
                    return {};
                }
                SearchState new_state = action.execute(current.state);
                if (visited.find(new_state) == visited.end()) {

                    AStarNode new_node = {compute_heuristic(new_state, *heuristic_), new_state,
                                          action, std::make_shared<AStarNode>(current)};
                    queue.push(new_node);
                }
            }
        }
    }
    return {};
}
