#include <boost/functional/hash.hpp>
#include <unordered_map>
#include <stdio.h>

template <typename Container> // we can make this generic for any container [1]
struct container_hash {
    std::size_t operator()(Container const& c) const {
        return boost::hash_range(c.begin(), c.end());
    }
};

typedef std::vector<int> intVector; // shorthand for the discrete state; maybe better to use floats
typedef std::unordered_map<intVector, int, container_hash<intVector>> statemap; // the dict
