#include<iostream>
#include<string>
#include<vector>
using namespace std;

#define MAX_LEN 50

// distance pour transformer str0 en str1 avec les opérations
// EDIT, REMOVE et ADD de coûts 1
unsigned edit_distance(string str0, string str1) {
	unsigned dist[MAX_LEN+1][MAX_LEN+1]; //dist[i][j] = d(str0[:i], str1[:j])
	size_t l0 = str0.size(), l1 = str1.size();

	for (unsigned j = 0; j <= l1; ++j)
		dist[0][j] = j;
	for (unsigned i = 1; i <= l0; ++i) {
		dist[i][0] = i;
		for (unsigned j = 1; j <= l1; ++j) {
			unsigned d = dist[i-1][j-1];
			if (str0[i-1] != str1[j-1]) ++d; // EDIT
			d = min(d, dist[i-1][j] + 1);    // REMOVE
			d = min(d, dist[i][j-1] + 1);    // ADD
			dist[i][j] = d;
		}
	}
	return dist[l0][l1];
}

int main(int argc, char* argv[]) {
	vector<pair<int, string>> words;
	while (cin.good()) {
		unsigned id;
		cin >> id;

		while (cin.good() && isspace(cin.get()));
		cin.unget();

		string s;
		getline(cin, s);
		if (!s.empty())
			words.push_back(pair<unsigned, string>(id, s));
	}

	for (unsigned i = 0; i < words.size(); ++i)
		for (unsigned j = 0; j < i; ++j) {
			unsigned dist = edit_distance(words[i].second, words[j].second);
			if (dist <= 2)
				cout << words[i].first << " " << words[j].first
					 << " " << dist << endl;
		}

	return 0;
}
