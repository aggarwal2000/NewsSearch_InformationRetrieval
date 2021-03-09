#include<iostream>
#include<string>
#include<vector>
#include<cstring>
#include<unordered_set>
#include<fstream>
#include<stdlib.h>
#include<regex>
#include <unordered_map> 
#include <unordered_set>
#include <utility>
#include <omp.h>
#include<time.h>
#include <iomanip>  


std::vector<std::vector<std::string>> Tokenizer(const std::string& doc_name)
{
    std::string line;
    std::ifstream fin;
    fin.open(doc_name);
    if(!fin)
       std::cout << "Could not open " << doc_name << std::endl;
    std::vector<std::string> tokens_title;
	std::vector<std::string> tokens_content;
	std::vector<std::string> tokens_author;
	std::vector<std::string> tokens_publication;

	std::vector<std::vector<std::string>> TOKENS;

	while(std::getline(fin,line) && line != std::string("publication:"))
	{
		if(line == "" || line == "title:" )
			continue;

		std::regex re("[\\|\n\t,:;(){.} ]+|[-]");
        std::sregex_token_iterator first{line.begin(), line.end(), re, -1}, last;//the '-1' is what makes the regex split (-1 := what was not matched)
        std::vector<std::string> tokens_temp{first, last}; //might conatain few empty strings bcoz of regex splitting...
		for(int i = 0;i<tokens_temp.size();i++)
			{
				if(tokens_temp[i] == std::string(""))
				   continue;
			    tokens_title.push_back(tokens_temp[i]);
			}
	}

	while(std::getline(fin,line) && line != std::string("author:"))
	{
		if(line == "")
			continue;

		std::regex re("[\\|\n\t,:;(){.} ]+|[-]");
        std::sregex_token_iterator first{line.begin(), line.end(), re, -1}, last;//the '-1' is what makes the regex split (-1 := what was not matched)
        std::vector<std::string> tokens_temp{first, last}; //might conatain few empty strings bcoz of regex splitting...
		for(int i = 0;i<tokens_temp.size();i++)
			{
				if(tokens_temp[i] == std::string(""))
				   continue;
			    tokens_publication.push_back(tokens_temp[i]);
			}
       

	}

	while(std::getline(fin,line) && line != std::string("date:"))
	{
		if(line == "")
			continue;

		std::regex re("[\\|\n\t,:;(){.} ]+|[-]");
        std::sregex_token_iterator first{line.begin(), line.end(), re, -1}, last;//the '-1' is what makes the regex split (-1 := what was not matched)
        std::vector<std::string> tokens_temp{first, last}; //might conatain few empty strings bcoz of regex splitting...
		for(int i = 0;i<tokens_temp.size();i++)
			{
				if(tokens_temp[i] == std::string(""))
				   continue;
			    tokens_author.push_back(tokens_temp[i]);
			}
        //tokens_author.insert(tokens_author.end(),tokens_temp.begin(),tokens_temp.end());

	}

	while(std::getline(fin,line) && line != std::string("content:"))
	{
	}

	while(std::getline(fin,line))
	{
		if(line == "")
			continue;

		std::regex re("[\\|\n\t,:;(){.} ]+|[-]");
        std::sregex_token_iterator first{line.begin(), line.end(), re, -1}, last;//the '-1' is what makes the regex split (-1 := what was not matched)
        std::vector<std::string> tokens_temp{first, last}; //might conatain few empty strings bcoz of regex splitting...
        //tokens_content.insert(tokens_content.end(),tokens_temp.begin(),tokens_temp.end());
		for(int i = 0;i<tokens_temp.size();i++)
			{
				if(tokens_temp[i] == std::string(""))
				   continue;
			    tokens_content.push_back(tokens_temp[i]);
			}

	}

    fin.close();

	TOKENS.push_back(tokens_title);
	TOKENS.push_back(tokens_publication);
	TOKENS.push_back(tokens_author);
	TOKENS.push_back(tokens_content);

    return TOKENS;
    
}


