import html.parser
from ._typing import OutCallback
from .elements import AnchorElement, ListElement
from _typeshed import Incomplete

__version__: Incomplete

class HTML2Text(html.parser.HTMLParser):
    split_next_td: bool
    td_count: int
    table_start: bool
    unicode_snob: Incomplete
    escape_snob: Incomplete
    escape_backslash: Incomplete
    escape_dot: Incomplete
    escape_plus: Incomplete
    escape_dash: Incomplete
    links_each_paragraph: Incomplete
    body_width: Incomplete
    skip_internal_links: Incomplete
    inline_links: Incomplete
    protect_links: Incomplete
    google_list_indent: Incomplete
    ignore_links: Incomplete
    ignore_mailto_links: Incomplete
    ignore_images: Incomplete
    images_as_html: Incomplete
    images_to_alt: Incomplete
    images_with_size: Incomplete
    ignore_emphasis: Incomplete
    bypass_tables: Incomplete
    ignore_tables: Incomplete
    google_doc: bool
    ul_item_mark: str
    emphasis_mark: str
    strong_mark: str
    single_line_break: Incomplete
    use_automatic_links: Incomplete
    hide_strikethrough: bool
    mark_code: Incomplete
    wrap_list_items: Incomplete
    wrap_links: Incomplete
    wrap_tables: Incomplete
    pad_tables: Incomplete
    default_image_alt: Incomplete
    tag_callback: Incomplete
    open_quote: Incomplete
    close_quote: Incomplete
    include_sup_sub: Incomplete
    out: Incomplete
    outtextlist: list[str]
    quiet: int
    p_p: int
    outcount: int
    start: bool
    space: bool
    a: list[AnchorElement]
    astack: list[dict[str, str | None] | None]
    maybe_automatic_link: str | None
    empty_link: bool
    absolute_url_matcher: Incomplete
    acount: int
    list: list[ListElement]
    blockquote: int
    pre: bool
    startpre: bool
    code: bool
    quote: bool
    br_toggle: str
    lastWasNL: bool
    lastWasList: bool
    style: int
    style_def: dict[str, dict[str, str]]
    tag_stack: list[tuple[str, dict[str, str | None], dict[str, str]]]
    emphasis: int
    drop_white_space: int
    inheader: bool
    abbr_title: str | None
    abbr_data: str | None
    abbr_list: dict[str, str]
    baseurl: Incomplete
    stressed: bool
    preceding_stressed: bool
    preceding_data: str
    current_tag: str
    def __init__(self, out: OutCallback | None = None, baseurl: str = '', bodywidth: int = ...) -> None: ...
    def update_params(self, **kwargs) -> None: ...
    def feed(self, data: str) -> None: ...
    def handle(self, data: str) -> str: ...
    def outtextf(self, s: str) -> None: ...
    def finish(self) -> str: ...
    def handle_charref(self, c: str) -> None: ...
    def handle_entityref(self, c: str) -> None: ...
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None: ...
    def handle_endtag(self, tag: str) -> None: ...
    def previousIndex(self, attrs: dict[str, str | None]) -> int | None: ...
    def handle_emphasis(self, start: bool, tag_style: dict[str, str], parent_style: dict[str, str]) -> None: ...
    inside_link: bool
    def handle_tag(self, tag: str, attrs: dict[str, str | None], start: bool) -> None: ...
    def pbr(self) -> None: ...
    def p(self) -> None: ...
    def soft_br(self) -> None: ...
    def o(self, data: str, puredata: bool = False, force: bool | str = False) -> None: ...
    def handle_data(self, data: str, entity_char: bool = False) -> None: ...
    def charref(self, name: str) -> str: ...
    def entityref(self, c: str) -> str: ...
    def google_nest_count(self, style: dict[str, str]) -> int: ...
    def optwrap(self, text: str) -> str: ...

def html2text(html: str, baseurl: str = '', bodywidth: int | None = None) -> str: ...

class CustomHTML2Text(HTML2Text):
    inside_pre: bool
    inside_code: bool
    inside_link: bool
    preserve_tags: Incomplete
    current_preserved_tag: Incomplete
    preserved_content: Incomplete
    preserve_depth: int
    handle_code_in_pre: Incomplete
    skip_internal_links: bool
    single_line_break: bool
    mark_code: bool
    include_sup_sub: bool
    body_width: int
    ignore_mailto_links: bool
    ignore_links: bool
    escape_backslash: bool
    escape_dot: bool
    escape_plus: bool
    escape_dash: bool
    escape_snob: bool
    def __init__(self, *args, handle_code_in_pre: bool = False, **kwargs) -> None: ...
    def update_params(self, **kwargs) -> None: ...
    def handle_tag(self, tag, attrs, start) -> None: ...
    def handle_data(self, data, entity_char: bool = False) -> None: ...
