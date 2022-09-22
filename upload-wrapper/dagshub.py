import requests
import urllib
import os
from pprint import pprint

DEFAULT_SOURCE_URL = "https://dagshub.com/"

if "BASE_URL" in os.environ:
	BASE_URL = os.environ['BASE_URL']
else:
	BASE_URL = "https://dagshub.com/"

CONTENT_UPLOAD_URL = "api/v1/repos/{owner}/{reponame}/content/main/{path}"

class Repo:
	src_url = DEFAULT_SOURCE_URL
	def __init__(self, owner, name, authToken=None, src_url=None):
		self.owner = owner
		self.name = name

		if authToken is not None:
			self.authToken = authToken
		elif "ACCESS_TOKEN" in os.environ:
			self.authToken = os.environ['ACCESS_TOKEN']
		else:
			raise Exception("Can't find access token. Please set enviroment variable ACCESS_TOKEN with a DagsHub access token")
		# TODO: verify token

		if src_url is not None:
			self.src_url = src_url

	def directory(self, path):
		return DataSet(self, path)

class Commit:
	choice="direct"
	message=""
	summary=""
	versioning="auto"
	new_branch=""
	def __init__(self):
		return

class DataSet:
	directory = ""
	files = []
	commit_data = Commit()
	def __init__(self, repo, directory):
		self.repo = repo
		self.directory = directory
		self.request_url = urllib.parse.urljoin(BASE_URL, CONTENT_UPLOAD_URL.format(
			owner=repo.owner,
			reponame=repo.name,
			path=urllib.parse.quote(directory, safe="")
		))

	def add(self, file, path="."):
		file_path = os.path.join(path, os.path.basename(os.path.normpath(file.name)))
		self.files.append((file_path, file))

	def commit(self, message, versioning=None, new_branch=None):
		data = {}
		if versioning is not None:
			self.commit_data.versioning = versioning
		data["versioning"] = self.commit_data.versioning

		if new_branch is not None:
			self.commit_data.choice = "commit-to-new-branch"
			self.commit_data.new_branch = new_branch

		data["commit_choice"] = self.commit_data.choice
		
		if self.commit_data.choice == "commit-to-new-branch":
			data["new_branch_name"] = self.commit_data.new_branch
		
		if message != "":
			self.commit_data.message = message
		else:
			raise Exception("You must provide a valid commit message")
		data["commit_message"] = self.commit_data.message

		# Prints for debugging
		print("URL: ", self.request_url)
		print("DATA:")
		pprint(data)
		print("Files:")
		pprint(self.files)
		print("making request...")
		res = requests.put(
			self.request_url, 
			data, 
			files=[("files", file) for file in self.files], 
			headers={'Authorization': 'token ' + self.repo.authToken})
		print("Response: ", res.status_code)
		pprint(res.content)