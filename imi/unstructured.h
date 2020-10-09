#include <boost/functional/hash.hpp>
#include <unordered_map>


template <typename Container> // we can make this generic for any container [1]
struct container_hash {
    std::size_t operator()(Container const& c) const {
        return boost::hash_range(c.begin(), c.end());
    }
};

typedef std::vector<int> intVector;
typedef std::unordered_map<intVector, int, container_hash<intVector>> statemap;
