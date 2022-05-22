#pragma once
#include <atomic>
#include <execution>
#include <functional>
#include <shared_mutex>
#include <mutex>
#include <optional>
#include <vector>
#include "Ranges.h"

// Provides thread-safe requesting and reporting of Tasks,
// aswell as progress metrics.
// Holds references to the Tasks.
template <typename Task>
class TaskLibrary
{
public:
	struct Task_Index { Task task; std::size_t index; };
	using value_type = std::reference_wrapper<Task>;
private:
	std::shared_mutex mutex;
	std::atomic<std::size_t> next = 0;
	std::atomic<std::size_t> processed = 0;
	std::vector<value_type> tasks;
public:
	TaskLibrary() noexcept = default;
	TaskLibrary(std::ranges::range auto& r) noexcept : tasks(r.begin(), r.end()) {}

	TaskLibrary(const TaskLibrary<Task>& o) noexcept : tasks(o.tasks) { next.store(o.next.load()); processed.store(o.processed.load()); }
	TaskLibrary(TaskLibrary<Task>&& o) noexcept : tasks(std::move(o.tasks)) { next.store(o.next.load()); processed.store(o.processed.load()); }
	TaskLibrary<Task>& operator=(const TaskLibrary<Task>& o) {
		if (this != &o)
			return *this;
		next.store(o.next.load());
		processed.store(o.processed.load());
		tasks = o.tasks;
		return *this;
	}
	TaskLibrary<Task>& operator=(TaskLibrary<Task>&& o) {
		if (this != &o)
			return *this;
		next.store(o.next.load());
		processed.store(o.processed.load());
		tasks = std::move(o.tasks);
		return *this;
	}
	~TaskLibrary() = default;

	template <typename ...Args>
	void emplace_back(Args&&... args) { tasks.emplace_back(std::forward<Args>(args)...); }
	void push_back(const Task& u) { tasks.push_back(u); }
	void push_back(Task&& u) { tasks.push_back(std::move(u)); }
	void reserve(std::size_t new_capacity) { tasks.reserve(new_capacity); }
	std::size_t size() const noexcept { return tasks.size(); }
	decltype(auto) front() const noexcept { return tasks.front(); }
	decltype(auto) back() const noexcept { return tasks.back(); }
	decltype(auto) begin() noexcept { return tasks.begin(); }
	decltype(auto) begin() const noexcept { return tasks.begin(); }
	decltype(auto) cbegin() const noexcept { return tasks.cbegin(); }
	decltype(auto) end() noexcept { return tasks.end(); }
	decltype(auto) end() const noexcept { return tasks.end(); }
	decltype(auto) cend() const noexcept { return tasks.cend(); }
	decltype(auto) operator[](std::size_t index) const noexcept { return tasks[index]; }
	decltype(auto) operator[](std::size_t index) noexcept { return tasks[index]; }

	// thread-safe
	std::size_t Scheduled() const noexcept { return next.load(std::memory_order_acquire); }
	std::size_t Processed() const noexcept { return processed.load(std::memory_order_acquire); }
	bool HasWork() const noexcept { return Scheduled() < tasks.size(); }
	bool IsDone() const noexcept { return Processed() == tasks.size(); }

	void lock() noexcept { mutex.lock(); }
	void unlock() noexcept { mutex.unlock(); }

	// thread-safe
	[[nodiscard]] std::optional<Task_Index> Request() {
		std::size_t index = next.fetch_add(1, std::memory_order_acq_rel);
		if (index < tasks.size())
			return Task_Index{ tasks[index].get(), index };
		return std::nullopt;
	}
	// thread-safe
	void Report(const Task& task, std::size_t index) {
		std::shared_lock lock(mutex);
		tasks[index].get() = task;
		processed.fetch_add(1, std::memory_order_release);
	}
	// thread-safe
	void Report(Task&& task, std::size_t index) {
		std::shared_lock lock(mutex);
		tasks[index].get() = std::move(task);
		processed.fetch_add(1, std::memory_order_release);
	}
	// thread-safe
	void Report(const Task_Index& ti) { Report(ti.task, ti.index); }
	// thread-safe
	void Report(Task_Index&& ti) { Report(std::move(ti.task), ti.index); }
};

template <typename Task> decltype(auto) begin(const TaskLibrary<Task>& p) { return p.begin(); }
template <typename Task> decltype(auto) begin(      TaskLibrary<Task>& p) { return p.begin(); }
template <typename Task> decltype(auto) end(const TaskLibrary<Task>& p) { return p.end(); }
template <typename Task> decltype(auto) end(      TaskLibrary<Task>& p) { return p.end(); }


template <typename Task>
class Executor
{
public:
	virtual void Process() = 0;
	virtual void LockData() = 0;
	virtual void UnlockData() = 0;
	// thread-safe
	virtual bool IsDone() const noexcept = 0;
	virtual std::size_t Scheduled() const noexcept = 0;
	virtual std::size_t Processed() const noexcept = 0;
	virtual std::size_t size() const noexcept = 0;
};

template <typename Task>
class SequentialExecutor final : public Executor<Task>
{
	std::mutex mutex;
	std::atomic<std::size_t> next = 0;
	std::function<void(Task&, std::size_t)> process;
	std::vector<std::reference_wrapper<Task>> tasks;
public:
	SequentialExecutor(range<Task> auto& tasks, std::invocable<Task&, std::size_t> auto process) noexcept
		: process(std::move(process))
		, tasks(tasks.begin(), tasks.end())
	{}

	void Process() override
	{
		for (; next < tasks.size(); next++)
		{
			Task copy = tasks[next].get();
			process(copy, next);
			std::scoped_lock lock(mutex);
			tasks[next].get() = copy;
		}
	}
	void LockData() override { mutex.lock(); }
	void UnlockData() override { mutex.unlock(); }
	bool IsDone() const noexcept override { return next == tasks.size(); }
	std::size_t Scheduled() const noexcept override { return next; }
	std::size_t Processed() const noexcept override { return std::max(std::size_t(0), next - 1); }
	std::size_t size() const noexcept override { return tasks.size(); }
};

template <typename Task>
class ParallelExecutor final : public Executor<Task>
{
	std::function<void(Task&, std::size_t)> process;
	TaskLibrary<Task> tasks;
public:
	ParallelExecutor(range<Task> auto& tasks, std::invocable<Task&, std::size_t> auto process) noexcept
		: process(std::move(process))
		, tasks(tasks)
	{}

	void Process() override
	{
		#pragma omp parallel
		while (true)
		{
			auto pair = tasks.Request();
			if (not pair.has_value())
				break;
			process(pair.value().task, pair.value().index);
			tasks.Report(pair.value());
		}
	}
	void LockData() override { tasks.lock(); }
	void UnlockData() override { tasks.unlock(); }
	bool IsDone() const noexcept override { return tasks.IsDone(); }
	std::size_t Scheduled() const noexcept override { return tasks.Scheduled(); }
	std::size_t Processed() const noexcept override { return tasks.Processed(); }
	std::size_t size() const noexcept override { return tasks.size(); }
};

//template <typename Task>
//class DistributedExecutor final : public Executor<Task>
//{
//public:
//};

template <std::ranges::range R, typename Task = std::ranges::range_value_t<R>>
std::unique_ptr<Executor<Task>> CreateExecutor(auto execution_policy, R&& tasks, std::invocable<Task&, std::size_t> auto process)
{
	if constexpr (std::is_same_v<decltype(execution_policy), std::execution::sequenced_policy>)
		return std::make_unique<SequentialExecutor<Task>>(tasks, process);
	if constexpr (std::is_same_v<decltype(execution_policy), std::execution::parallel_policy>)
		return std::make_unique<ParallelExecutor<Task>>(tasks, process);
}

template <std::ranges::range R, typename Task = std::ranges::range_value_t<R>>
auto CreateExecutor(auto execution_policy, R&& tasks, std::invocable<Task&> auto process)
{
	return CreateExecutor(execution_policy, tasks,
		[process](Task& task, std::size_t) { process(task); });
}

template <std::ranges::range R, typename Task = std::ranges::range_value_t<R>>
void Process(auto execution_policy, R&& tasks, std::invocable<Task&, std::size_t> auto process)
{
	CreateExecutor(execution_policy, tasks, process)->Process();
}

template <std::ranges::range R, typename Task = std::ranges::range_value_t<R>>
void Process(auto execution_policy, R&& tasks, std::invocable<Task&> auto process)
{
	Process(execution_policy, tasks, [process](Task& p, std::size_t) { process(p); });
}


template <typename Task>
class TaskLibraryGroup
{
public:
	struct Task_Indices { Task task; std::size_t group_index; std::size_t task_index; };
private:
	std::atomic<std::size_t> next = 0;
	std::atomic<std::size_t> processed = 0;
	std::vector<std::size_t> cum_sizes;
	std::vector<TaskLibrary<Task>> tls; // TaskLibraries

	std::size_t GroupIndex(std::size_t index) const { return std::distance(cum_sizes.begin(), std::upper_bound(cum_sizes.begin(), cum_sizes.end() - 1, index)); }
	void AddCumSize() { cum_sizes.push_back(tls.back().size() + (cum_sizes.empty() ? 0 : cum_sizes.back())); }
public:
	TaskLibraryGroup() noexcept = default;
	TaskLibraryGroup(nested_range auto&& r) noexcept : tls(r.begin(), r.end()) {
		for (const auto& t : tls)
			cum_sizes.push_back(t.size() + (cum_sizes.empty() ? 0 : cum_sizes.back()));
	}
	TaskLibraryGroup(const TaskLibraryGroup<Task>& o) noexcept : cum_sizes(o.cum_sizes), tls(o.tls) {
		next.store(o.next.load());
		processed.store(o.processed.load());
	}
	TaskLibraryGroup(TaskLibraryGroup<Task>&& o) noexcept : cum_sizes(std::move(o.cum_sizes)), tls(std::move(o.tls)) {
		next.store(o.next.load());
		processed.store(o.processed.load());
	}
	TaskLibraryGroup<Task>& operator=(const TaskLibraryGroup<Task>& o) {
		if (this != &o)
			return *this;
		next.store(o.next.load());
		processed.store(o.processed.load());
		cum_sizes = o.cum_sizes;
		tls = o.tls;
		return *this;
	}
	TaskLibraryGroup<Task>& operator=(TaskLibraryGroup<Task>&& o) {
		if (this != &o)
			return *this;
		next.store(o.next.load());
		processed.store(o.processed.load());
		cum_sizes = o.cum_sizes;
		tls = std::move(o.tls);
		return *this;
	}
	~TaskLibraryGroup() = default;

	template <typename ...Args>
	void emplace_back(Args&&... args) { tls.emplace_back(std::forward<Args>(args)...); AddCumSize(); }
	void push_back(const Task& u) { tls.push_back(u); AddCumSize(); }
	void push_back(Task&& u) { tls.push_back(std::move(u)); AddCumSize(); }
	void reserve(std::size_t new_capacity) { tls.reserve(new_capacity); cum_sizes.reserve(new_capacity); }
	std::size_t size() const { return tls.size(); }
	decltype(auto) front() const { return tls.front(); }
	decltype(auto) back() const { return tls.back(); }
	decltype(auto) begin() { return tls.begin(); }
	decltype(auto) begin() const { return tls.begin(); }
	decltype(auto) cbegin() const { return tls.cbegin(); }
	decltype(auto) end() { return tls.end(); }
	decltype(auto) end() const { return tls.end(); }
	decltype(auto) cend() const { return tls.cend(); }
	decltype(auto) operator[](std::size_t group_index) const { return tls[group_index]; }
	decltype(auto) operator[](std::size_t group_index) { return tls[group_index]; }

	// thread-safe
	std::size_t Scheduled() const noexcept { return next.load(std::memory_order_acquire); }
	std::size_t Processed() const noexcept { return processed.load(std::memory_order_acquire); }
	bool HasWork() const noexcept { return Scheduled() < tls.size(); }
	bool IsDone() const noexcept { return Processed() == tls.size(); }

	std::size_t Scheduled(std::size_t group_index) const { return tls[group_index].Scheduled(); }
	std::size_t Processed(std::size_t group_index) const { return tls[group_index].Processed(); }
	bool HasWork(std::size_t group_index) const { return tls[group_index].HasWork(); }
	bool IsDone(std::size_t group_index) const { return tls[group_index].IsDone(); }

	void lock(std::size_t group_index) noexcept { tls[group_index].lock(); }
	void unlock(std::size_t group_index) noexcept { tls[group_index].unlock(); }
	void lock() noexcept { for (auto& tl : tls) tl.lock(); }
	void unlock() noexcept { for (auto& tl : tls) tl.unlock(); }

	// thread-safe
	[[nodiscard]] std::optional<Task_Indices> Request() {
		std::size_t index = next.fetch_add(1, std::memory_order_acq_rel);
		if (cum_sizes.empty() or index >= cum_sizes.back())
			return std::nullopt;
		std::size_t group_index = GroupIndex(index);
		auto request = tls[group_index].Request();
		if (request.has_value())
			return Task_Indices{ std::move(request.value().task), group_index, request.value().index };
	}
	// thread-safe
	void Report(const Task& task, std::size_t group_index, std::size_t task_index) {
		tls[group_index].Report(task, task_index);
		processed.fetch_add(1, std::memory_order_release);
	}
	// thread-safe
	void Report(Task&& task, std::size_t group_index, std::size_t task_index) {
		tls[group_index].Report(std::move(task), task_index);
		processed.fetch_add(1, std::memory_order_release);
	}
	// thread-safe
	void Report(const Task_Indices& ti) { Report(ti.task, ti.group_index, ti.task_index); }
	// thread-safe
	void Report(Task_Indices&& ti) { Report(std::move(ti.task), ti.group_index, ti.task_index); }
};

template <typename Task>
class GroupExecutor
{
public:
	virtual void Process() = 0;
	virtual void LockData() = 0;
	virtual void UnlockData() = 0;
	virtual bool IsDone() const noexcept = 0;
	virtual bool IsDone(std::size_t group_index) const noexcept = 0;
};

template <typename Task>
class SequentialGroupExecutor final : public GroupExecutor<Task>
{
	std::mutex mutex;
	std::atomic<std::size_t> next = 0;
	std::function<void(Task&, std::size_t, std::size_t)> process;
	std::vector<std::vector<std::reference_wrapper<Task>>> tasks;
	std::vector<std::size_t> cum_sizes;
public:
	template <nested_range Range>
	SequentialGroupExecutor(Range&& tasks, std::function<void(Task&, std::size_t, std::size_t)> process) noexcept
		: process(std::move(process))
	{
		this->tasks.reserve(tasks.size());
		for (std::vector<Task>& t : tasks)
			this->tasks.emplace_back(t.begin(), t.end());

		cum_sizes.reserve(tasks.size());
		std::size_t sum = 0;
		for (std::vector<Task>& t : tasks)
		{
			sum += t.size();
			cum_sizes.push_back(sum);
		}
	}

	void Process() override
	{
		for (std::size_t i = 0; i < tasks.size(); i++)
			for (std::size_t j = 0; j < tasks[i].size(); j++, next++)
			{
				Task copy = tasks[i][j].get();
				process(copy, i, j);
				std::scoped_lock lock(mutex);
				tasks[i][j].get() = copy;
			}
	}
	void LockData() override { mutex.lock(); }
	void UnlockData() override { mutex.unlock(); }
	bool IsDone() const noexcept override { return next == cum_sizes.back(); }
	bool IsDone(std::size_t group_index) const noexcept override { return next >= cum_sizes[group_index]; }
};


template <typename Task>
class ParallelGroupExecutor final : public GroupExecutor<Task>
{
	std::function<void(Task&, std::size_t, std::size_t)> process;
	TaskLibraryGroup<Task> tlgs;
public:
	template <nested_range Range>
	ParallelGroupExecutor(Range&& tasks, std::function<void(Task&, std::size_t, std::size_t)> process) noexcept
		: process(std::move(process)), tlgs(tasks) {}

	void Process() override
	{
		#pragma omp parallel
		while (true)
		{
			auto triple = tlgs.Request();
			if (not triple.has_value())
				break;
			process(triple.value().task, triple.value().group_index, triple.value().task_index);
			tlgs.Report(std::move(triple.value()));
		}
	}
	void LockData() override { tlgs.lock(); }
	void UnlockData() override { tlgs.unlock(); }
	bool IsDone() const noexcept override { return tlgs.IsDone(); }
	bool IsDone(std::size_t group_index) const noexcept override { return tlgs.IsDone(group_index); }
};

template <nested_range R, typename Task = std::ranges::range_value_t<std::ranges::range_value_t<R>>>
std::unique_ptr<Executor<Task>> CreateExecutor(auto execution_policy, R&& tasks, std::invocable<Task&, std::size_t, std::size_t> auto process)
{
	if constexpr (std::is_same_v<decltype(execution_policy), std::execution::sequenced_policy>)
		return std::make_unique<SequentialGroupExecutor<Task>>(tasks, process);
	if constexpr (std::is_same_v<decltype(execution_policy), std::execution::parallel_policy>)
		return std::make_unique<ParallelGroupExecutor<Task>>(tasks, process);
}

template <nested_range R, typename Task = std::ranges::range_value_t<std::ranges::range_value_t<R>>>
auto CreateExecutor(auto execution_policy, R&& tasks, std::invocable<Task&> auto process)
{
	return CreateExecutor(execution_policy, tasks, [process](Task& task, std::size_t, std::size_t) { process(task); });
}
