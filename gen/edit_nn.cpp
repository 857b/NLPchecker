#include<iostream>
#include<string>
#include<locale>
#include<codecvt>
#include<vector>
#include<fstream>
#include<unordered_map>
using namespace std;

#define MAX_LEN 50

struct hash_pair {
	size_t operator()(const pair<char32_t, char32_t> p) const {
		return (((size_t)p.first) << 16) + p.second;
	}
};
unordered_map<pair<char32_t, char32_t>, float, hash_pair> char_cost;

// distance pour transformer str0 en str1 avec les opérations
// REMOVE et ADD de coûts 1
// EDIT de coût variable (char_conf)
float edit_distance(u32string str0, u32string str1) {
	float dist[MAX_LEN+1][MAX_LEN+1]; //dist[i][j] = d(str0[:i], str1[:j])
	size_t l0 = str0.size(), l1 = str1.size();

	for (unsigned j = 0; j <= l1; ++j)
		dist[0][j] = j;
	for (unsigned i = 1; i <= l0; ++i) {
		dist[i][0] = i;
		for (unsigned j = 1; j <= l1; ++j) {
			float d = dist[i-1][j-1];
			if (str0[i-1] != str1[j-1]) {    // EDI
				auto cost = char_cost.find(pair<char32_t, char32_t>
								(str0[i-1], str1[j-1]));
				if (cost != char_cost.end())
					d += cost->second;
				else
					++d;
			}
			d = min(d, dist[i-1][j] + 1);    // REMOVE
			d = min(d, dist[i][j-1] + 1);    // ADD
			dist[i][j] = d;
		}
	}
	return dist[l0][l1];
}

int main(int argc, char* argv[]) {
	vector<pair<int, u32string>> words;
	wstring_convert<codecvt_utf8<char32_t>, char32_t> conv;

	if (argc > 1) {
		ifstream char_confus(argv[1]);
		if (!char_confus)
			return 1;
		while (char_confus.good()) {
			float c;
			string s;
			char_confus >> c >> s;
			if (!s.empty()) {
				u32string s32 = conv.from_bytes(s);
				for (unsigned i = 0; i < s32.size(); ++i)
					for (unsigned j = 0; j < s32.size(); ++j)
						if (i != j)
							char_cost[pair<char32_t, char32_t>(s32[i], s32[j])]
								= c;
			}
		}
	}

	while (cin.good()) {
		unsigned id;
		cin >> id;

		while (cin.good() && isspace(cin.get()));
		cin.unget();

		string s;
		getline(cin, s);
		if (!s.empty())
			words.push_back(pair<unsigned, u32string>(id, conv.from_bytes(s)));
	}

	for (unsigned i = 0; i < words.size(); ++i)
		for (unsigned j = 0; j < i; ++j) {
			float dist = edit_distance(words[i].second, words[j].second);
			if (dist <= 2)
				cout << words[i].first << " " << words[j].first
					 << " " << dist << endl;
		}

	return 0;
}
