public:
    // types
    typedef Key key_type;
    typedef Element mapped_type;
    typedef std::pair<const Key, Element> value_type;
    typedef Hasher hasher;
    typedef Equality key_equal;
    typedef Allocator allocator_type;
    typedef typename allocator_type::pointer pointer;
    typedef typename allocator_type::const_pointer const_pointer;
    typedef typename allocator_type::reference reference;
    typedef typename allocator_type::const_reference const_reference;
    typedef implementation-defined size_type;
    typedef implementation-defined difference_type;
    typedef implementation-defined iterator;
    typedef implementation-defined const_iterator;
    typedef implementation-defined local_iterator;
    typedef implementation-defined const_local_iterator;

    allocator_type get_allocator() const;

    // size and capacity
    bool empty() const;     // May take linear time!
    size_type size() const; // May take linear time!
    size_type max_size() const;

    // iterators
    iterator begin();
    const_iterator begin() const;
    iterator end();
    const_iterator end() const;
    const_iterator cbegin() const;
    const_iterator cend() const;

    // modifiers
    std::pair<iterator, bool> insert(const value_type& x);
    iterator insert(const_iterator hint, const value_type& x);
    template<class InputIterator>
    void insert(InputIterator first, InputIterator last);
    // Supported since C++11
    std::pair<iterator, bool> insert(value_type&& x);
    iterator insert(const_iterator hint, value_type&& x);
    void insert(std::initializer_list<value_type> il);
    // Supported since C++11
    template<typename... Args>
    std::pair<iterator, bool> emplace(Args&&... args);
    template<typename... Args>
    iterator emplace_hint(const_iterator hint, Args&&... args);

    iterator unsafe_erase(const_iterator position);
    size_type unsafe_erase(const key_type& k);
    iterator unsafe_erase(const_iterator first, const_iterator last);
    void clear();

    // observers
    hasher hash_function() const;
    key_equal key_eq() const;

    // lookup
    iterator find(const key_type& k);
    const_iterator find(const key_type& k) const;
    size_type count(const key_type& k) const;
    std::pair<iterator, iterator> equal_range(const key_type& k);
    std::pair<const_iterator, const_iterator> equal_range(const key_type& k) const;

    // parallel iteration
    typedef implementation-defined range_type;
    typedef implementation-defined const_range_type;
    range_type range();
    const_range_type range() const;

    // bucket interface - for debugging
    size_type unsafe_bucket_count() const;
    size_type unsafe_max_bucket_count() const;
    size_type unsafe_bucket_size(size_type n);
    size_type unsafe_bucket(const key_type& k) const;
    local_iterator unsafe_begin(size_type n);
    const_local_iterator unsafe_begin(size_type n) const;
    local_iterator unsafe_end(size_type n);
    const_local_iterator unsafe_end(size_type n) const;
    const_local_iterator unsafe_cbegin(size_type n) const;
    const_local_iterator unsafe_cend(size_type n) const;

    // hash policy
    float load_factor() const;
    float max_load_factor() const;
    void max_load_factor(float z);
    void rehash(size_type n);

public:
    // construct/destroy/copy
    explicit concurrent_unordered_map(size_type n = implementation-defined,
    const hasher& hf = hasher(),
    const key_equal& eql = key_equal(),
    const allocator_type& a = allocator_type());
    concurrent_unordered_map(size_type n, const allocator_type& a);
    concurrent_unordered_map(size_type n, const hasher& hf, const allocator_type& a);
    template <typename InputIterator>
    concurrent_unordered_map(InputIterator first, InputIterator last,
                              size_type n = implementation-defined,
                              const hasher& hf = hasher(),
                              const key_equal& eql = key_equal(),
                              const allocator_type& a = allocator_type());
    template <typename InputIterator>
    concurrent_unordered_map(InputIterator first, InputIterator last,
                              size_type n, const allocator_type& a);
    template <typename InputIterator>
    concurrent_unordered_map(InputIterator first, InputIterator last,
                              size_type n, const hasher& hf, const allocator_type& a);
    concurrent_unordered_map(const concurrent_unordered_map&);
    explicit concurrent_unordered_map(const allocator_type&);
    concurrent_unordered_map(const concurrent_unordered_map&, const allocator_type&);
    // Supported since C++11
    concurrent_unordered_map(concurrent_unordered_map&&);
    concurrent_unordered_map(concurrent_unordered_map&&, const allocator_type&);
    concurrent_unordered_map(std::initializer_list<value_type> il,
                              size_type n = implementation-defined,
                              const hasher& hf = hasher(),
                              const key_equal& eql = key_equal(),
                              const allocator_type& a = allocator_type());
    concurrent_unordered_map(std::initializer_list<value_type> il,
                              size_type n, const allocator_type& a);
    concurrent_unordered_map(std::initializer_list<value_type> il,
                              size_type n, const hasher& hf, const allocator_type& a);

    ~concurrent_unordered_map();

    concurrent_unordered_map& operator=(const concurrent_unordered_map&);
    // Supported since C++11
    concurrent_unordered_map& operator=(concurrent_unordered_map&&);
    concurrent_unordered_map& operator=(std::initializer_list<value_type> il);

    void swap(concurrent_unordered_map&);

    mapped_type& operator[](const key_type& k);
    mapped_type& at( const key_type& k );
    const mapped_type& at(const key_type& k) const;
