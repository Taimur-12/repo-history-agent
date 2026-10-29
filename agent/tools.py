from ingest import github
from agent.trace import traced

search = traced(github.search)
get_pr = traced(github.get_pr)
get_issue = traced(github.get_issue)
get_timeline = traced(github.get_timeline)
get_comments = traced(github.get_comments)
version_for_pr = traced(github.version_for_pr)
file_history = traced(github.file_history)

TOOLS = {
    "search": search,
    "get_pr": get_pr,
    "get_issue": get_issue,
    "get_timeline": get_timeline,
    "get_comments": get_comments,
    "version_for_pr": version_for_pr,
    "file_history": file_history,
}
