use anyhow::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap},
    Frame, Terminal,
};
use serde::{Deserialize, Serialize};
use std::io;
use syntect::easy::HighlightLines;
use syntect::highlighting::{ThemeSet, Style as SyntectStyle};
use syntect::parsing::SyntaxSet;
use syntect::util::LinesWithEndings;

mod context_client;
use context_client::ContextClient;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SearchResult {
    id: String,
    document_id: String,
    title: String,
    summary: String,
    score: f64,
    url: String,
    has_summary: bool,
    code_snippets_count: usize,
    code_snippet_ids: Vec<String>,
    content_type: String,
    chunk_index: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SearchResponse {
    results: Vec<SearchResult>,
    total: usize,
    query: String,
    mode: String,
    execution_time_ms: u64,
    note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CodeSnippet {
    id: String,
    document_id: String,
    content: String,
    language: String,
    start_line: Option<usize>,
    end_line: Option<usize>,
    metadata: serde_json::Value,
}

#[derive(Default)]
struct AppState {
    search_query: String,
    search_results: Vec<SearchResult>,
    selected_result: usize,
    selected_snippet: usize,
    current_code_snippet: Option<CodeSnippet>,
    code_snippets: Vec<CodeSnippet>,
    input_mode: InputMode,
    list_state: ListState,
    search_status: String,
    context_client: ContextClient,
    context_name: String,
}

#[derive(Default, PartialEq)]
enum InputMode {
    #[default]
    Normal,
    Editing,
}

impl AppState {
    fn new() -> Self {
        let mut state = Self::default();
        state.list_state.select(Some(0));
        
        // Get server URL from environment or use default
        let server_url = std::env::var("CONTEXT_SERVER_URL")
            .unwrap_or_else(|_| "http://localhost:8000".to_string());
        state.context_client = ContextClient::new(&server_url);
        
        // Pre-fill search query if provided
        if let Ok(query) = std::env::var("DEFAULT_QUERY") {
            state.search_query = query;
        }
        
        // Set context name from environment or use default
        state.context_name = std::env::var("DEFAULT_CONTEXT")
            .unwrap_or_else(|_| "test".to_string());
        
        state
    }

    async fn search(&mut self) -> Result<()> {
        if self.search_query.is_empty() {
            return Ok(());
        }

        self.search_status = "Searching...".to_string();
        
        match self.context_client.search(&self.context_name, &self.search_query, 10).await {
            Ok(response) => {
                self.search_results = response.results;
                self.selected_result = 0;
                self.list_state.select(Some(0));
                self.search_status = format!("Found {} results", self.search_results.len());
                
                // Load first code snippet if available
                if !self.search_results.is_empty() && !self.search_results[0].code_snippet_ids.is_empty() {
                    let snippet_id = self.search_results[0].code_snippet_ids[0].clone();
                    self.load_code_snippet(&snippet_id).await?;
                }
            }
            Err(e) => {
                self.search_status = format!("Search failed: {}", e);
            }
        }
        
        Ok(())
    }

    async fn load_code_snippet(&mut self, snippet_id: &str) -> Result<()> {
        match self.context_client.get_code_snippet(&self.context_name, snippet_id).await {
            Ok(snippet) => {
                self.current_code_snippet = Some(snippet);
            }
            Err(e) => {
                self.search_status = format!("Failed to load snippet: {}", e);
            }
        }
        Ok(())
    }

    async fn load_all_snippets_for_result(&mut self, result_index: usize) -> Result<()> {
        if let Some(result) = self.search_results.get(result_index) {
            self.code_snippets.clear();
            
            for snippet_id in &result.code_snippet_ids {
                match self.context_client.get_code_snippet(&self.context_name, snippet_id).await {
                    Ok(snippet) => {
                        self.code_snippets.push(snippet);
                    }
                    Err(e) => {
                        self.search_status = format!("Failed to load snippet {}: {}", snippet_id, e);
                    }
                }
            }
            
            if !self.code_snippets.is_empty() {
                self.selected_snippet = 0;
                self.current_code_snippet = Some(self.code_snippets[0].clone());
            }
        }
        Ok(())
    }

    fn next_result(&mut self) {
        if !self.search_results.is_empty() {
            self.selected_result = (self.selected_result + 1) % self.search_results.len();
            self.list_state.select(Some(self.selected_result));
        }
    }

    fn previous_result(&mut self) {
        if !self.search_results.is_empty() {
            if self.selected_result == 0 {
                self.selected_result = self.search_results.len() - 1;
            } else {
                self.selected_result -= 1;
            }
            self.list_state.select(Some(self.selected_result));
        }
    }

    fn next_snippet(&mut self) {
        if !self.code_snippets.is_empty() {
            self.selected_snippet = (self.selected_snippet + 1) % self.code_snippets.len();
            self.current_code_snippet = Some(self.code_snippets[self.selected_snippet].clone());
        }
    }

    fn previous_snippet(&mut self) {
        if !self.code_snippets.is_empty() {
            if self.selected_snippet == 0 {
                self.selected_snippet = self.code_snippets.len() - 1;
            } else {
                self.selected_snippet -= 1;
            }
            self.current_code_snippet = Some(self.code_snippets[self.selected_snippet].clone());
        }
    }
}

fn main() -> Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app state
    let mut app = AppState::new();
    
    // If a query was pre-filled, perform initial search
    let should_auto_search = !app.search_query.is_empty();

    // Setup syntax highlighting
    let syntax_set = SyntaxSet::load_defaults_newlines();
    let theme_set = ThemeSet::load_defaults();
    let theme = &theme_set.themes["base16-ocean.dark"];

    let rt = tokio::runtime::Runtime::new()?;
    let result = rt.block_on(run_app(&mut terminal, &mut app, &syntax_set, theme, should_auto_search));

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = result {
        println!("{err:?}");
    }

    Ok(())
}

async fn run_app(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut AppState,
    syntax_set: &SyntaxSet,
    theme: &syntect::highlighting::Theme,
    should_auto_search: bool,
) -> Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    
    // Perform initial search if query was pre-filled
    if should_auto_search {
        let _ = app.search().await;
    }
    
    loop {
        terminal.draw(|f| ui(f, app, syntax_set, theme))?;

        if let Event::Key(key) = event::read()? {
            if key.kind == KeyEventKind::Press {
                match app.input_mode {
                    InputMode::Normal => match key.code {
                        KeyCode::Char('q') => return Ok(()),
                        KeyCode::Char('s') => {
                            app.input_mode = InputMode::Editing;
                        }
                        KeyCode::Down | KeyCode::Char('j') => {
                            app.next_result();
                            if let Some(result) = app.search_results.get(app.selected_result) {
                                if !result.code_snippet_ids.is_empty() {
                                    let _ = rt.block_on(app.load_all_snippets_for_result(app.selected_result));
                                }
                            }
                        }
                        KeyCode::Up | KeyCode::Char('k') => {
                            app.previous_result();
                            if let Some(result) = app.search_results.get(app.selected_result) {
                                if !result.code_snippet_ids.is_empty() {
                                    let _ = rt.block_on(app.load_all_snippets_for_result(app.selected_result));
                                }
                            }
                        }
                        KeyCode::Right | KeyCode::Char('l') => {
                            app.next_snippet();
                        }
                        KeyCode::Left | KeyCode::Char('h') => {
                            app.previous_snippet();
                        }
                        _ => {}
                    },
                    InputMode::Editing => match key.code {
                        KeyCode::Enter => {
                            let _ = rt.block_on(app.search());
                            app.input_mode = InputMode::Normal;
                        }
                        KeyCode::Char(c) => {
                            app.search_query.push(c);
                        }
                        KeyCode::Backspace => {
                            app.search_query.pop();
                        }
                        KeyCode::Esc => {
                            app.input_mode = InputMode::Normal;
                        }
                        _ => {}
                    },
                }
            }
        }
    }
}

fn ui(f: &mut Frame, app: &AppState, syntax_set: &SyntaxSet, theme: &syntect::highlighting::Theme) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(0),
            Constraint::Length(3),
        ])
        .split(f.area());

    let input_style = match app.input_mode {
        InputMode::Normal => Style::default(),
        InputMode::Editing => Style::default().fg(Color::Yellow),
    };

    let search_title = format!(
        "Search in '{}' (s to edit, Enter to search, q to quit)", 
        app.context_name
    );
    let search_input = Paragraph::new(app.search_query.as_str())
        .style(input_style)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(search_title)
        );
    f.render_widget(search_input, chunks[0]);

    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
        .split(chunks[1]);

    // Search results
    let items: Vec<ListItem> = app
        .search_results
        .iter()
        .enumerate()
        .map(|(i, result)| {
            let content = format!(
                "{} (Score: {:.3}) [{} snippets]",
                result.title,
                result.score,
                result.code_snippets_count
            );
            
            let style = if i == app.selected_result {
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            
            ListItem::new(content).style(style)
        })
        .collect();

    let results_list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Search Results (↑↓ to navigate)")
        )
        .highlight_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD));

    f.render_stateful_widget(results_list, main_chunks[0], &mut app.list_state.clone());

    // Code snippet viewer
    let code_content = if let Some(ref snippet) = app.current_code_snippet {
        let lines = highlight_code(&snippet.content, &snippet.language, syntax_set, theme);
        Text::from(lines)
    } else {
        Text::from("No code snippet selected")
    };

    let snippet_title = if let Some(ref snippet) = app.current_code_snippet {
        format!("Code Snippet ({}) [←→ to navigate snippets]", snippet.language)
    } else {
        "Code Snippet".to_string()
    };

    let code_viewer = Paragraph::new(code_content)
        .block(Block::default().borders(Borders::ALL).title(snippet_title))
        .wrap(Wrap { trim: false });

    f.render_widget(code_viewer, main_chunks[1]);

    // Status bar
    let status = Paragraph::new(app.search_status.as_str())
        .style(Style::default().fg(Color::Green))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL).title("Status"));
    f.render_widget(status, chunks[2]);
}

fn highlight_code<'a>(
    code: &'a str,
    language: &str,
    syntax_set: &SyntaxSet,
    theme: &syntect::highlighting::Theme,
) -> Vec<Line<'a>> {
    let syntax = syntax_set
        .find_syntax_by_extension(language)
        .or_else(|| syntax_set.find_syntax_by_name(language))
        .unwrap_or_else(|| syntax_set.find_syntax_plain_text());

    let mut highlighter = HighlightLines::new(syntax, theme);
    let mut lines = Vec::new();

    for line in LinesWithEndings::from(code) {
        let highlighted = highlighter.highlight_line(line, syntax_set).unwrap();
        let mut spans = Vec::new();
        
        for (style, text) in highlighted {
            let color = style_to_color(style);
            spans.push(Span::styled(text.to_string(), Style::default().fg(color)));
        }
        
        lines.push(Line::from(spans));
    }

    lines
}

fn style_to_color(style: SyntectStyle) -> Color {
    Color::Rgb(style.foreground.r, style.foreground.g, style.foreground.b)
}
