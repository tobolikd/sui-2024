#include "search-strategies.h"
#include "memusage.h"
#include <queue>
#include <set>

const size_t memory_limit = 49000000;

std::vector<SearchAction> BreadthFirstSearch::solve(const SearchState &init_state) {
	std::queue<std::pair<SearchState, std::vector<SearchAction>>> queue;
	std::set<SearchState> visited;

	queue.push({init_state,{}});

	while(!queue.empty()){
		if(getCurrentRSS()>memory_limit){
			return {};
		}
		auto [actual_state, actual_path]=queue.front();
		queue.pop();

		if(actual_state.isFinal()){
			return actual_path;
		}

		if(visited.find(actual_state)==visited.end()){
			visited.insert(actual_state);

			for(auto &action : actual_state.actions()){
				SearchState new_state = action.execute(actual_state);
				if(visited.find(new_state)==visited.end()){
					auto new_path=actual_path;
					new_path.push_back(action);
					queue.push({new_state,new_path});
				}
			}
		}

	}
	return {};
}

std::vector<SearchAction> DepthFirstSearch::solve(const SearchState &init_state) {
	return {};
}

double StudentHeuristic::distanceLowerBound(const GameState &state) const {
    return 0;
}

std::vector<SearchAction> AStarSearch::solve(const SearchState &init_state) {
	return {};
}
