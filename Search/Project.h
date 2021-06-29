#pragma once
#include <atomic>
#include <execution>
#include <functional>
#include <shared_mutex>
#include <optional>
#include <vector>
#include <thread>
#include <ranges>

// Allows thread-safe requesting and reporting of Tasks,
// aswell as getting progress metrics.
// Holds references to the Tasks.
template <typename Task>
class Project
{
public:
	struct Task_Index { Task task; std::size_t index; };
	using value_type = std::reference_wrapper<Task>;
private:
	mutable std::shared_mutex mutex;
	std::atomic<std::size_t> next = 0;
	std::atomic<std::size_t> processed = 0;
	std::vector<value_type> tasks;
public:
	Project() noexcept = default;
	template <typename Iterator>
	Project(Iterator first, Iterator last) noexcept : tasks(first, last) {}
	Project(std::vector<std::reference_wrapper<Task>> tasks) noexcept : tasks(std::move(tasks)) {}

	Project(const Project<Task>& o) noexcept : tasks(o.tasks) { next.store(o.next.load()); processed.store(o.processed.load()); }
	Project(Project<Task>&& o) noexcept : tasks(std::move(o.tasks)) { next.store(o.next.load()); processed.store(o.processed.load()); }
	Project<Task>& operator=(const Project<Task>& o) {
		if (this != &o)
			return *this;
		next.store(o.next.load());
		processed.store(o.processed.load());
		tasks = o.tasks;
		return *this;
	}
	Project<Task>& operator=(Project<Task>&& o) {
		if (this != &o)
			return *this;
		next.store(o.next.load());
		processed.store(o.processed.load());
		tasks = std::move(o.tasks);
		return *this;
	}
	~Project() = default;

	template <typename ...Args>
	void emplace_back(Args&&... args) { tasks.emplace_back(std::forward<Args>(args)...); }
	void push_back(const Task& u) { tasks.push_back(u); }
	void push_back(Task&& u) { tasks.push_back(std::move(u)); }
	void reserve(std::size_t new_capacity) { tasks.reserve(new_capacity); }
	[[nodiscard]] std::size_t size() const { return tasks.size(); }
	[[nodiscard]] decltype(auto) front() const { return tasks.front(); }
	[[nodiscard]] decltype(auto) back() const { return tasks.back(); }
	[[nodiscard]] decltype(auto) begin() { return tasks.begin(); }
	[[nodiscard]] decltype(auto) begin() const { return tasks.begin(); }
	[[nodiscard]] decltype(auto) cbegin() const { return tasks.cbegin(); }
	[[nodiscard]] decltype(auto) end() { return tasks.end(); }
	[[nodiscard]] decltype(auto) end() const { return tasks.end(); }
	[[nodiscard]] decltype(auto) cend() const { return tasks.cend(); }
	[[nodiscard]] decltype(auto) operator[](std::size_t index) const { return tasks[index]; }
	[[nodiscard]] decltype(auto) operator[](std::size_t index) { return tasks[index]; }

	// thread-safe
	[[nodiscard]] std::size_t Scheduled() const noexcept { return next.load(std::memory_order_acquire); }
	[[nodiscard]] std::size_t Processed() const noexcept { return processed.load(std::memory_order_acquire); }
	[[nodiscard]] bool HasWork() const noexcept { return Scheduled() < tasks.size(); }
	[[nodiscard]] bool IsDone() const noexcept { return Processed() == tasks.size(); }

	void lock() const noexcept { mutex.lock(); }
	void unlock() const noexcept { mutex.unlock(); }

	// thread-safe
	[[nodiscard]] std::optional<Task_Index> Request() {
		std::size_t index = next.fetch_add(1, std::memory_order_acq_rel);
		if (index < tasks.size())
			return Task_Index{ tasks[index].get(), index };
		return std::nullopt;
	}
	// thread-safe
	void Report(const Task& task, std::size_t index) {
		mutex.lock_shared();
		tasks[index].get() = task;
		mutex.unlock_shared();
		processed.fetch_add(1, std::memory_order_release);
	}
	// thread-safe
	void Report(Task&& task, std::size_t index) {
		mutex.lock_shared();
		tasks[index].get() = std::move(task);
		mutex.unlock_shared();
		processed.fetch_add(1, std::memory_order_release);
	}
	// thread-safe
	void Report(const Task_Index& ti) { Report(ti.task, ti.index); }
	// thread-safe
	void Report(Task_Index&& ti) { Report(std::move(ti.task), ti.index); }
};

template <typename Task> decltype(auto) begin(const Project<Task>& p) { return p.begin(); }
template <typename Task> decltype(auto) begin(      Project<Task>& p) { return p.begin(); }
template <typename Task> decltype(auto) end(const Project<Task>& p) { return p.end(); }
template <typename Task> decltype(auto) end(      Project<Task>& p) { return p.end(); }

template <typename Task>
class ProjectGroup
{
public:
	struct Task_Indices { Task task; std::size_t group_index; std::size_t task_index; };
private:
	std::atomic<std::size_t> next = 0;
	std::atomic<std::size_t> processed = 0;
	std::vector<std::size_t> cumulative_size; // {groups[0].size(), groups[0].size() + groups[1].size(), ...}
	std::vector<Project<Task>> groups;
	std::size_t GroupIndex(int index) const { return std::distance(cumulative_size.begin(), std::upper_bound(cumulative_size.begin(), cumulative_size.end(), index)); }
	void UpdateCumulativeSize()
	{
		cumulative_size.clear();
		std::size_t sum = 0;
		for (const auto& project : groups)
		{
			sum += project.size();
			cumulative_size.push_back(sum);
		}
		cumulative_size.pop_back();
	}
public:
	ProjectGroup() noexcept = default;
	ProjectGroup(const ProjectGroup<Task>&o) noexcept : cumulative_size(o.cumulative_size), groups(o.groups) { next.store(o.next.load()); processed.store(o.processed.load()); }
	ProjectGroup(ProjectGroup<Task> && o) noexcept : cumulative_size(std::move(o.cumulative_size)), groups(std::move(o.groups)) { next.store(o.next.load()); processed.store(o.processed.load()); }
	ProjectGroup<Task>& operator=(const ProjectGroup<Task>&o) {
		if (this != &o)
			return *this;
		next.store(o.next.load());
		processed.store(o.processed.load());
		cumulative_size = o.cumulative_size;
		groups = o.groups;
		return *this;
	}
	ProjectGroup<Task>& operator=(ProjectGroup<Task> && o) {
		if (this != &o)
			return *this;
		next.store(o.next.load());
		processed.store(o.processed.load());
		cumulative_size = o.cumulative_size;
		groups = std::move(o.groups);
		return *this;
	}
	~ProjectGroup() = default;

	template <typename ...Args>
	void emplace_back(Args&&... args) { groups.emplace_back(std::forward<Args>(args)...); UpdateCumulativeSize(); }
	void push_back(const Task& u) { groups.push_back(u); UpdateCumulativeSize(); }
	void push_back(Task&& u) { groups.push_back(std::move(u)); UpdateCumulativeSize(); }
	void reserve(std::size_t new_capacity) { groups.reserve(new_capacity); }
	[[nodiscard]] std::size_t size() const { return groups.size(); }
	[[nodiscard]] decltype(auto) front() const { return groups.front(); }
	[[nodiscard]] decltype(auto) back() const { return groups.back(); }
	[[nodiscard]] decltype(auto) begin() { return groups.begin(); }
	[[nodiscard]] decltype(auto) begin() const { return groups.begin(); }
	[[nodiscard]] decltype(auto) cbegin() const { return groups.cbegin(); }
	[[nodiscard]] decltype(auto) end() { return groups.end(); }
	[[nodiscard]] decltype(auto) end() const { return groups.end(); }
	[[nodiscard]] decltype(auto) cend() const { return groups.cend(); }
	[[nodiscard]] decltype(auto) operator[](std::size_t index) const { return groups[index]; }
	[[nodiscard]] decltype(auto) operator[](std::size_t index) { return groups[index]; }

	// thread-safe
	[[nodiscard]] std::size_t Scheduled() const noexcept { return next.load(std::memory_order_acquire); }
	[[nodiscard]] std::size_t Processed() const noexcept { return processed.load(std::memory_order_acquire); }
	[[nodiscard]] bool HasWork() const noexcept { return Scheduled() < groups.size(); }
	[[nodiscard]] bool IsDone() const noexcept { return Processed() == groups.size(); }

	[[nodiscard]] std::size_t Scheduled(int index) const { return groups[index].Scheduled(); }
	[[nodiscard]] std::size_t Processed(int index) const { return groups[index].Processed(); }
	[[nodiscard]] bool HasWork(int index) const { return groups[index].HasWork(); }
	[[nodiscard]] bool IsDone(int index) const { return groups[index].IsDone(); }

	void lock(int index) const noexcept { groups[index].lock(); }
	void unlock(int index) const noexcept { groups[index].unlock(); }

	// thread-safe
	[[nodiscard]] std::optional<Task_Indices> Request() {
		std::size_t index = next.fetch_add(1, std::memory_order_acq_rel);
		if (index < cumulative_size.back()) {
			std::size_t group_index = GroupIndex(index);
			auto request = groups[group_index].Request();
			if (request.has_value())
				return Task_Indices{ std::move(request.task), group_index, request.index };
		}
		return std::nullopt;
	}
	// thread-safe
	void Report(const Task& task, std::size_t group_index, std::size_t task_index) {
		groups[group_index].Report(task, task_index);
		processed.fetch_add(1, std::memory_order_release);
	}
	// thread-safe
	void Report(Task&& task, std::size_t group_index, std::size_t task_index) {
		groups[group_index].Report(std::move(task), task_index);
		processed.fetch_add(1, std::memory_order_release);
	}
	// thread-safe
	void Report(const Task_Indices& ti) { Report(ti.tasks, ti.group_index, ti.task_index); }
	// thread-safe
	void Report(Task_Indices&& ti) { Report(std::move(ti.tasks), ti.group_index, ti.task_index); }
};

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

//template <typename Task>
//class Executor
//{
//public:
//	virtual void Start() = 0;
//	virtual void Stop() = 0;
//	virtual void Pause() = 0;
//};
//
//template <typename Task>
//class SequentialExecutorVec final : public Executor<Task>
//{
//	std::function<void(Task&)> process;
//public:
//	SequentialExecutor(std::function<void(Task&)> process) noexcept : Executor(std::move(process)) {}
//
//	void Start() override;
//	void Stop() override;
//	void Pause() override;
//};
//
//template <typename Task>
//class ParallelExecutor final : public Executor<Task>
//{
//public:
//};
//
//template <typename Task>
//class DistributedExecutor final : public Executor<Task>
//{
//public:
//};
//
//template <typename Task, typename ExecutionPolicy>
//std::unique_ptr<Executor<Task>> CreateExecutor(ExecutionPolicy, std::vector<Task>&, std::function<void(std::optional<Task_Index>)> process)
//{
//	if constexpr (std::is_same_v<ExecutionPolicy, std::execution::sequenced_policy>)
//	{
//	}
//	if constexpr (std::is_same_v<ExecutionPolicy, std::execution::parallel_policy>)
//	{
//	}
//}
//
//template <typename Task, typename ExecutionPolicy>
//std::unique_ptr<Executor<Task>> CreateExecutor(ExecutionPolicy, std::vector<std::vector<Task>>&, std::function<void(Task&)> process)
//{
//	if constexpr (std::is_same_v<ExecutionPolicy, std::execution::sequenced_policy>)
//	{
//	}
//	if constexpr (std::is_same_v<ExecutionPolicy, std::execution::parallel_policy>)
//	{
//	}
//}

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
				process(pair.value().task, pair.value().index);
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
//			process(pair.value().task, pair.value().index);
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