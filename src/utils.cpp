#include "utils.h"


std::vector<std::array<int, 2>> get_group_indices(veci groups) 
{
    // assumed the group indeces are ordered
    std::vector<std::array<int, 2>> gi;
    
    int start = 0;
    int end = 0;
    int g = groups(0);

    for (int i = 0; i < groups.size(); i++) {
	if (groups(i) != g) {
	    end = i - 1;
	    gi.push_back(std::array{start, end});

	    g = groups(i);
	    start = i;
	}
    }
    int p = groups.rows() - 1;
    gi.push_back({start, p});

    return gi;
}


double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}
