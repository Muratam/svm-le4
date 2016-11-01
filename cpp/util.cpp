#include "util.h"

unordered_map<string, string> parse_args(
    vector<string> args, vector<std::pair<string, string>> kvs) {
  unordered_map<string, string> res;
  for (auto kv : kvs) {
    auto seekedarg = ALL(std::find, args, std::get<0>(kv));
    if (seekedarg != args.end()) {
      res[std::get<0>(kv)] =
          seekedarg + 1 == args.end() ? std::get<1>(kv) : *(seekedarg + 1);
    }
  }
  return res;
}
