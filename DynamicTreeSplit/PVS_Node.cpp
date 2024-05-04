#include "PVS_Node.h"

std::string to_string(NodeType type)
{
    switch (type) {
    case NodeType::PV: return "PV";
    case NodeType::All: return "All";
    case NodeType::Cut: return "Cut";
    default: return "Unknown";
    }
}

std::string to_string(NodeStatus status)
{
    switch (status) {
    case NodeStatus::Unsolved: return "Unsolved";
    case NodeStatus::Blocked: return "Blocked";
    case NodeStatus::Solved: return "Solved";
    default: return "Unknown";
    }
}
