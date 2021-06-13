#pragma once
#include <atomic>
#include <execution>
#include <functional>
#include <shared_mutex>
#include <optional>
#include <vector>
#include <thread>
#include <ranges>

// Allows thread-safe requesting and reporting of work-units,
// aswell as progress metrics.
// Holds references to the work-units.
template <typename T>
class Project
{
public:
	struct WU_Index { T wu; std::size_t index; };
	mutable std::shared_mutex mutex;
private:
	std::atomic<std::size_t> next = 0;
	std::atomic<std::size_t> processed = 0;
	std::vector<std::reference_wrapper<T>> wu;
public:

	Project() noexcept = default;
	template <typename Iterator>
	Project(Iterator first, Iterator last) noexcept : wu(first, last) {}
	Project(std::vector<std::reference_wrapper<T>> wu) noexcept : wu(std::move(wu)) {}

	Project(const Project<T>& o) noexcept : wu(o.wu) { next.store(o.next.load()); processed.store(o.processed.load()); }
	Project(Project<T>&& o) noexcept : wu(std::move(o.wu)) { next.store(o.next.load()); processed.store(o.processed.load()); }
	Project<T>& operator=(const Project<T>& o) {
		if (this != &o)
			return *this;
		next.store(o.next.load());
		processed.store(o.processed.load());
		wu = o.wu;
		return *this;
	}
	Project<T>& operator=(Project<T>&& o) {
		if (this != &o)
			return *this;
		next.store(o.next.load());
		processed.store(o.processed.load());
		wu = std::move(o.wu);
		return *this;
	}
	~Project() = default;

	using value_type = std::reference_wrapper<T>; // needed for 'back_insert_iterator'.

	template <typename ...Args>
	void emplace_back(Args&&... args) { wu.emplace_back(std::forward<Args>(args)...); }
	void push_back(const T& u) { wu.push_back(u); }
	void push_back(T&& u) { wu.push_back(std::move(u)); }
	void reserve(std::size_t new_capacity) { wu.reserve(new_capacity); }
	[[nodiscard]] std::size_t size() const { return wu.size(); }
	[[nodiscard]] decltype(auto) front() const { return wu.front(); }
	[[nodiscard]] decltype(auto) back() const { return wu.back(); }
	[[nodiscard]] decltype(auto) begin() { return wu.begin(); }
	[[nodiscard]] decltype(auto) begin() const { return wu.begin(); }
	[[nodiscard]] decltype(auto) cbegin() const { return wu.cbegin(); }
	[[nodiscard]] decltype(auto) end() { return wu.end(); }
	[[nodiscard]] decltype(auto) end() const { return wu.end(); }
	[[nodiscard]] decltype(auto) cend() const { return wu.cend(); }
	[[nodiscard]] decltype(auto) operator[](std::size_t index) const { return wu[index]; }
	[[nodiscard]] decltype(auto) operator[](std::size_t index) { return wu[index]; }

	// thread-safe
	[[nodiscard]] std::size_t Scheduled() const { return next.load(std::memory_order_acquire); }
	[[nodiscard]] std::size_t Processed() const { return processed.load(std::memory_order_acquire); }
	[[nodiscard]] bool HasWork() const { return Scheduled() < wu.size(); }
	[[nodiscard]] bool IsDone() const { return Processed() == wu.size(); }

	// thread-safe
	[[nodiscard]] std::optional<WU_Index> Request() {
		std::size_t index = next.fetch_add(1, std::memory_order_acq_rel);
		if (index < wu.size())
			return WU_Index{ wu[index].get(), index };
		return std::nullopt;
	}
	// thread-safe
	void Report(const T& u, std::size_t index) {
		mutex.lock_shared();
		wu[index].get() = u;
		mutex.unlock_shared();
		processed.fetch_add(1, std::memory_order_release);
	}
	// thread-safe
	void Report(T&& u, std::size_t index) {
		mutex.lock_shared();
		wu[index].get() = std::move(u);
		mutex.unlock_shared();
		processed.fetch_add(1, std::memory_order_release);
	}
	// thread-safe
	void Report(const WU_Index& u) { Report(u.wu, u.index); }
	// thread-safe
	void Report(WU_Index&& u) { Report(std::move(u.wu), u.index); }
};

template <typename T> auto&& begin(const Project<T>& p) { return p.begin(); }
template <typename T> auto&& begin(      Project<T>& p) { return p.begin(); }
template <typename T> auto&& end(const Project<T>& p) { return p.end(); }
template <typename T> auto&& end(      Project<T>& p) { return p.end(); }


//template <typename T>
//class Projects
//{
//	std::vector<Project<T>> proj;
//public:
//	Projects() noexcept = default;
//	template <typename Iterator>
//	Projects(const Iterator& begin, const Iterator& end) noexcept : proj(begin, end) {}
//	Projects(std::vector<Project<T>> proj) noexcept : proj(std::move(proj)) {}
//
//	using value_type = T; // needed for 'back_insert_iterator'.
//
//	template <typename ...Args>
//	void emplace_back(Args&&... args) { proj.emplace_back(std::forward<Args>(args)...); }
//	void push_back(const Project<T>& p) { proj.push_back(p); }
//	void push_back(Project<T>&& p) { proj.push_back(std::move(p)); }
//	void reserve(std::size_t new_capacity) { proj.reserve(new_capacity); }
//	[[nodiscard]] std::size_t size() const { return proj.size(); }
//	[[nodiscard]] decltype(auto) front() const { return proj.front(); }
//	[[nodiscard]] decltype(auto) back() const { return proj.back(); }
//	[[nodiscard]] decltype(auto) begin() { return proj.begin(); }
//	[[nodiscard]] decltype(auto) begin() const { return proj.begin(); }
//	[[nodiscard]] decltype(auto) cbegin() const { return proj.cbegin(); }
//	[[nodiscard]] decltype(auto) end() { return proj.end(); }
//	[[nodiscard]] decltype(auto) end() const { return proj.end(); }
//	[[nodiscard]] decltype(auto) cend() const { return proj.cend(); }
//	[[nodiscard]] decltype(auto) operator[](std::size_t index) const { return proj[index]; }
//	[[nodiscard]] decltype(auto) operator[](std::size_t index) { return proj[index]; }
//};

template <typename Iterable, typename ExecutionPolicy>
void Process(ExecutionPolicy,
	Iterable& iterable,
	std::function<void(typename Iterable::value_type&, std::size_t)> process)
{
	if constexpr (std::is_same_v<ExecutionPolicy, std::execution::sequenced_policy>)
	{
		auto it = std::begin(iterable);
		auto end = std::end(iterable);
		for (std::size_t i = 0; it != end; ++it)
			process(*it, i++);
	}
	if constexpr (std::is_same_v<ExecutionPolicy, std::execution::parallel_policy>)
	{
		Project<typename Iterable::value_type> project(std::begin(iterable), std::end(iterable));
		#pragma omp parallel
		{
			while (true)
			{
				auto pair = project.Request();
				if (not pair.has_value())
					break;
				process(pair.value().wu, pair.value().index);
				project.Report(std::move(pair.value()));
			}
		}
	}
}

template <typename Iterable, typename ExecutionPolicy>
void Process(ExecutionPolicy&& expo,
	Iterable& iterable,
	std::function<void(typename Iterable::value_type&)> process)
{
	Process(expo, iterable, [&process](typename Iterable::value_type& p, std::size_t) { process(p); });
}

//template <typename Project, typename ExecutionPolicy>
//void Process(ExecutionPolicy,
//	Project& project,
//	std::function<void(typename Project::value_type&, std::size_t)> process)
//{
//	#pragma omp parallel if(std::is_same_v<ExecutionPolicy, std::execution::parallel_policy>)
//	{
//		while (true)
//		{
//			auto pair = project.Request();
//			if (not pair.has_value())
//				break;
//			process(pair.value().wu, pair.value().index);
//			project.Report(std::move(pair.value()));
//		}
//	}
//}
//
//template <typename Project, typename ExecutionPolicy>
//void Process(ExecutionPolicy&& expo,
//	Project& project,
//	std::function<void(typename Project::value_type&)> process)
//{
//	Process(expo, project, [&process](typename Project::value_type& p, std::size_t) { process(p); });
//}
//
//template <typename Project, typename ExecutionPolicy>
//void Process(ExecutionPolicy,
//	std::vector<Project>& projects, 
//	std::function<void(typename Project::value_type&, std::size_t)> process,
//	std::function<void(const Project&)> project_done_task = nullptr)
//{
//	#pragma omp parallel if(std::is_same_v<ExecutionPolicy, std::execution::parallel_policy>)
//	{
//		for (Project& proj : projects)
//		{
//			Process(std::execution::seq, proj, process);
//			if (project_done_task != nullptr)
//				if (proj.IsDone())
//					project_done_task(proj);
//		}
//	}
//}
//
//template <typename Project, typename ExecutionPolicy>
//void Process(ExecutionPolicy expo,
//	std::vector<Project>& projects,
//	std::function<void(typename Project::value_type&)> process,
//	std::function<void(const Project&)> project_done_task = nullptr)
//{
//	Process(expo, projects, [&process](typename Project::value_type& p, std::size_t) { process(p); }, project_done_task);
//}